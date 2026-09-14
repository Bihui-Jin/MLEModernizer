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

0.75676

# 6. Current score

0.50501

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Diagnosis: Cell 1 crashes because it tries to read Kaggle “../input/…” submission files that do not exist in this environment (only `/kaggle/data` and related folders are available). The notebook expects six submission CSVs from other Kaggle notebooks/datasets, but those paths are unavailable here, causing a `FileNotFoundError` on the first `pd.read_csv`.

Patch summary: Modify cell 1 only to load a valid in-environment CSV as a deterministic fallback when the external ensemble submission files are missing. To preserve downstream variable names and interfaces, create `data1`..`data6` as DataFrames with the expected `id` and `target` columns by reading the provided `sample_submission.csv` from an existing path.

Updated cells: (cell 1 only)

Compatibility notes for cell k+1: `data1` remains a pandas DataFrame and has `.head()` available; it also contains `id` and `target` columns consistent with typical submission format.

Assumptions: The intended next steps only require `data1`..`data6` to exist with `id`/`target` columns (not necessarily the original ensemble predictions), and `/kaggle/data/sample_submission.csv` (or its nested equivalent) is present as listed.'
- What this solution (achieved 0.51432) has done: 'The crash happens because `features_from_snippet()` assumes each `.npy` snippet has 4 dimensions after selecting panels (so it can reduce over axes `(1,2,3)`), but in this environment the loaded arrays are 3D, making axis `3` invalid. The minimal fix is to compute `A_panel_means` by reducing over *all* non-panel axes dynamically, so it works for both 3D and 4D snippet shapes while keeping the same intended statistic (mean per A panel). This change is localized to cell 1 and preserves all downstream variables (`feat_df`, `pred1..pred6`, `data1..data6`) with identical meanings. No other logic, paths, feature definitions, or scoring is changed.'
- What this solution (achieved 0.50817) has done: 'The timeout is dominated by Python-level loops repeatedly loading thousands of `.npy` files and doing relatively expensive feature steps (notably `np.quantile`) one file at a time, plus a second full pass over up to 20k training files for stacking. I keep the exact feature definitions and scoring/stacking logic, but speed up data access and feature extraction by (1) reading only the required IDs (avoid scanning/merging overhead), (2) parallelizing `.npy` loads + feature computation with a thread pool (I/O + NumPy release the GIL), and (3) making feature math more allocation-efficient (views, fewer reshapes/copies) while remaining numerically equivalent. I also ensure we don’t compute any unused training features when stacking is disabled, and I keep determinism stable (fixed ordering, no randomness). These changes reduce wall time substantially without changing the model logic, features, or evaluation semantics.'
- What this solution (achieved 0.50501) has done: 'Your current AUC (0.508) is far below the target (0.75676), so we need a real lift while keeping the same feature-and-logistic-stacking core. The biggest minimal-impact issue is that the stacking model is trained on **only the first 20k training IDs in sorted order**, which can be distributionally biased; switching to a deterministic but better-mixed subset across all training files usually improves generalization without changing the model/feature logic. I keep the exact same features, transforms, and LogisticRegression, but select the 20k training samples via a fixed-seed shuffle for representativeness and then scale the stacking inputs using train-set mean/std (still a linear transform of the same six base predictions). Finally, I ensure test predictions are reindexed to `sample_submission.csv` order to avoid any accidental misalignment.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)



## === cell 1
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

DATA_ROOT_CANDIDATES = [
    "/kaggle/data",  # this environment
    "/kaggle/data/seti-breakthrough-listen",
    "/kaggle/input/seti-breakthrough-listen",  # original Kaggle path (if present)
    "/kaggle/input",  # fallback
]


def find_existing_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


DATA_ROOT = find_existing_path(DATA_ROOT_CANDIDATES)
if DATA_ROOT is None:
    raise FileNotFoundError(f"None of the data roots exist: {DATA_ROOT_CANDIDATES}")

if os.path.exists("/kaggle/data/seti-breakthrough-listen"):
    BASE = "/kaggle/data/seti-breakthrough-listen"
else:
    BASE = "/kaggle/data"

SAMPLE_SUB_PATH_CANDS = [
    os.path.join(BASE, "sample_submission.csv"),
    "/kaggle/data/sample_submission.csv",
    "/kaggle/data/seti-breakthrough-listen/sample_submission.csv",
    "../input/sample_submission.csv",
]
sample_path = next((p for p in SAMPLE_SUB_PATH_CANDS if os.path.exists(p)), None)
if sample_path is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv in: " + ", ".join(SAMPLE_SUB_PATH_CANDS)
    )

TEST_DIR_CANDS = [
    os.path.join(BASE, "test"),
    "/kaggle/data/test",
    "/kaggle/data/seti-breakthrough-listen/test",
]
test_dir = next((p for p in TEST_DIR_CANDS if os.path.isdir(p)), None)
if test_dir is None:
    raise FileNotFoundError(
        "Could not find test directory in: " + ", ".join(TEST_DIR_CANDS)
    )

TRAIN_DIR_CANDS = [
    os.path.join(BASE, "train"),
    "/kaggle/data/train",
    "/kaggle/data/seti-breakthrough-listen/train",
]
train_dir = next((p for p in TRAIN_DIR_CANDS if os.path.isdir(p)), None)
if train_dir is None:
    raise FileNotFoundError(
        "Could not find train directory in: " + ", ".join(TRAIN_DIR_CANDS)
    )

TRAIN_LABELS_CANDS = [
    os.path.join(BASE, "train_labels.csv"),
    "/kaggle/data/train_labels.csv",
    "/kaggle/data/seti-breakthrough-listen/train_labels.csv",
]
train_labels_path = next((p for p in TRAIN_LABELS_CANDS if os.path.exists(p)), None)
if train_labels_path is None:
    raise FileNotFoundError(
        "Could not find train_labels.csv in: " + ", ".join(TRAIN_LABELS_CANDS)
    )

sample = pd.read_csv(sample_path)
if not {"id", "target"}.issubset(sample.columns):
    raise ValueError(
        f"sample_submission.csv must contain id,target. Got columns: {sample.columns.tolist()}"
    )

train_labels = pd.read_csv(train_labels_path)
if not {"id", "target"}.issubset(train_labels.columns):
    raise ValueError(
        f"train_labels.csv must contain id,target. Got columns: {train_labels.columns.tolist()}"
    )


def build_id_to_path_map(root: str):
    id2p = {}
    rp = Path(root)
    for sub in rp.glob("*"):
        if sub.is_dir():
            for p in sub.glob("*.npy"):
                sid = p.stem
                if sid not in id2p:
                    id2p[sid] = p
    for p in rp.glob("*.npy"):
        sid = p.stem
        if sid not in id2p:
            id2p[sid] = p
    return id2p


A_IDX = (0, 2, 4)
O_IDX = (1, 3, 5)


def robust_scale(x, eps=1e-6):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + eps
    return (x - med) / mad


def sigmoid(z):
    z = np.clip(z, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-z))


def features_from_snippet(arr):
    x = arr.astype(np.float32, copy=False)

    A = x[list(A_IDX)]
    O = x[list(O_IDX)]

    A_abs = float(np.mean(np.abs(A)))
    O_abs = float(np.mean(np.abs(O)))
    A_sq = float(np.mean(A * A))
    O_sq = float(np.mean(O * O))

    diff_mean = np.mean(A, axis=0) - np.mean(O, axis=0)
    diff_sq = float(np.mean(diff_mean * diff_mean))

    def grad_mag(imgs):
        dt = np.diff(imgs, axis=1)
        df = np.diff(imgs, axis=2)
        return float(np.mean(np.abs(dt)) + np.mean(np.abs(df)))

    A_grad = grad_mag(A)
    O_grad = grad_mag(O)

    A_p99 = float(np.quantile(A, 0.99))
    O_p99 = float(np.quantile(O, 0.99))

    A_panel_means = np.mean(A, axis=tuple(range(1, A.ndim)))
    A_consistency = float(-np.std(A_panel_means))  # higher is "more consistent"

    return {
        "A_abs": A_abs,
        "O_abs": O_abs,
        "A_sq": A_sq,
        "O_sq": O_sq,
        "diff_sq": diff_sq,
        "A_grad": A_grad,
        "O_grad": O_grad,
        "A_p99": A_p99,
        "O_p99": O_p99,
        "A_consistency": A_consistency,
    }


def compute_feat_df_from_ids(ids, id2path, max_workers=None):
    ids = list(ids)

    def _one(sid):
        p = id2path.get(sid)
        if p is None:
            return None
        arr = np.load(p, allow_pickle=False)
        feats = features_from_snippet(arr)
        feats["id"] = sid
        return feats

    if max_workers is None:
        cpu = os.cpu_count() or 4
        max_workers = min(8, max(2, cpu))

    rows = [None] * len(ids)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, feats in enumerate(ex.map(_one, ids, chunksize=32)):
            rows[i] = feats

    missing = [ids[i] for i, r in enumerate(rows) if r is None]
    if missing:
        raise FileNotFoundError(
            f"Missing .npy for {len(missing)} ids (showing up to 5): {missing[:5]}"
        )
    return pd.DataFrame(rows)


def make_pred(series_score, center_to=0.0, scale=1.0, bias=0.0):
    s = series_score.to_numpy(dtype=np.float32)
    s = robust_scale(s)  # makes mapping more stable
    z = (s - center_to) * scale + bias
    return sigmoid(z).astype(np.float64)


test_id2path = build_id_to_path_map(test_dir)
test_ids = sample["id"].astype(str).tolist()
feat_df = compute_feat_df_from_ids(test_ids, test_id2path)
feat_df = feat_df.set_index("id").reindex(test_ids).reset_index()

score1 = np.log1p(feat_df["A_sq"]) - np.log1p(feat_df["O_sq"])
pred1 = make_pred(score1, scale=1.5)

score2 = np.log1p(feat_df["diff_sq"])
pred2 = make_pred(score2, scale=1.2)

score3 = np.log1p(feat_df["A_grad"]) - np.log1p(feat_df["O_grad"])
pred3 = make_pred(score3, scale=1.3)

score4 = feat_df["A_p99"] - feat_df["O_p99"]
pred4 = make_pred(score4, scale=1.0)

score5 = (
    np.log1p(feat_df["diff_sq"])
    + 0.5 * (np.log1p(feat_df["A_sq"]) - np.log1p(feat_df["O_sq"]))
    + 0.3 * (np.log1p(feat_df["A_grad"]) - np.log1p(feat_df["O_grad"]))
    + 0.2 * feat_df["A_consistency"]
)
pred5 = make_pred(score5, scale=1.0)

score6 = feat_df["A_consistency"]
pred6 = make_pred(score6, scale=0.8)

data1 = pd.DataFrame({"id": feat_df["id"].values, "target": pred1})
data2 = pd.DataFrame({"id": feat_df["id"].values, "target": pred2})
data3 = pd.DataFrame({"id": feat_df["id"].values, "target": pred3})
data4 = pd.DataFrame({"id": feat_df["id"].values, "target": pred4})
data5 = pd.DataFrame({"id": feat_df["id"].values, "target": pred5})
data6 = pd.DataFrame({"id": feat_df["id"].values, "target": pred6})

try:
    from sklearn.linear_model import LogisticRegression
except Exception as e:
    LogisticRegression = None
    _sk_import_error = e

use_stacking = LogisticRegression is not None

if use_stacking:
    MAX_TRAIN_N = 20000  # unchanged
    train_id2path = build_id_to_path_map(train_dir)

    rng = np.random.RandomState(1337)
    all_train_ids = np.array(sorted(train_id2path.keys()), dtype=object)
    rng.shuffle(all_train_ids)
    train_ids_used = all_train_ids[: min(MAX_TRAIN_N, len(all_train_ids))].tolist()

    train_feat = compute_feat_df_from_ids(train_ids_used, train_id2path)
    train_feat = train_labels.merge(
        train_feat, on="id", how="inner", validate="one_to_one"
    )

    tr_score1 = np.log1p(train_feat["A_sq"]) - np.log1p(train_feat["O_sq"])
    tr_pred1 = make_pred(tr_score1, scale=1.5)

    tr_score2 = np.log1p(train_feat["diff_sq"])
    tr_pred2 = make_pred(tr_score2, scale=1.2)

    tr_score3 = np.log1p(train_feat["A_grad"]) - np.log1p(train_feat["O_grad"])
    tr_pred3 = make_pred(tr_score3, scale=1.3)

    tr_score4 = train_feat["A_p99"] - train_feat["O_p99"]
    tr_pred4 = make_pred(tr_score4, scale=1.0)

    tr_score5 = (
        np.log1p(train_feat["diff_sq"])
        + 0.5 * (np.log1p(train_feat["A_sq"]) - np.log1p(train_feat["O_sq"]))
        + 0.3 * (np.log1p(train_feat["A_grad"]) - np.log1p(train_feat["O_grad"]))
        + 0.2 * train_feat["A_consistency"]
    )
    tr_pred5 = make_pred(tr_score5, scale=1.0)

    tr_score6 = train_feat["A_consistency"]
    tr_pred6 = make_pred(tr_score6, scale=0.8)

    X_tr = np.vstack(
        [tr_pred1, tr_pred2, tr_pred3, tr_pred4, tr_pred5, tr_pred6]
    ).T.astype(np.float32)
    y_tr = train_feat["target"].to_numpy(dtype=np.int32)

    mu = X_tr.mean(axis=0, keepdims=True)
    sd = X_tr.std(axis=0, keepdims=True) + 1e-6
    X_tr_s = (X_tr - mu) / sd

    lr = LogisticRegression(
        solver="lbfgs",
        max_iter=300,
        n_jobs=None,
        C=1.0,
    )
    lr.fit(X_tr_s, y_tr)

    X_te = np.vstack([pred1, pred2, pred3, pred4, pred5, pred6]).T.astype(np.float32)
    X_te_s = (X_te - mu) / sd
    stacked_pred = lr.predict_proba(X_te_s)[:, 1].astype(np.float64)

    data_stack = pd.DataFrame({"id": feat_df["id"].values, "target": stacked_pred})
else:
    data_stack = None



## === cell 2
data1.head()



## === cell 3
data2.head()



## === cell 4
if data_stack is not None:
    data6 = data_stack.copy()
else:
    data6 = data1.copy()
    data6["target"] = (
        0.75 * data5["target"]
        + 0.12 * data4["target"]
        + 0.12 * data6["target"]
        + 0.005 * data2["target"]
        + 0.005 * data3["target"]
    )



## === cell 5
sub = sample[["id"]].merge(
    data6[["id", "target"]], on="id", how="left", validate="one_to_one"
)
if sub["target"].isna().any():
    raise ValueError(
        "Some test ids are missing predictions after merge; cannot write valid submission."
    )
sub.to_csv("submission.csv", index=False)
print(
    "Wrote submission.csv with",
    len(sub),
    "rows. target min/mean/max:",
    float(sub["target"].min()),
    float(sub["target"].mean()),
    float(sub["target"].max()),
)
if data_stack is None:
    print("Stacking disabled (sklearn import failed).")
