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

0.7627048215298181

# 6. Current score

0.50274

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Your notebook fails because it tries to read several external “../input/…” submission files that don’t exist in this Kaggle environment, so none of the ensemble dataframes are created. To make it run end-to-end and still follow the same “weighted blend of submissions” core logic, I keep the blending step but generate each input submission from the provided `sample_submission.csv` as a safe fallback (uniform 0.5 predictions), then validate column names/order and write a proper `submission.csv`. This yield a valid submission file reliably; since there is no current score yet, this prioritizes correctness and producing a submission over optimizing AUC.'
- What this solution (achieved 0.50274) has done: 'The timeout is dominated by loading ~60k `.npy` files one-by-one and repeatedly running expensive `np.partition` three times per file for quantiles. I keep the exact feature definitions and model unchanged, but speed up feature extraction by computing the 10/50/90% quantiles in a single `np.partition` call per sample (mathematically identical to your linear interpolation), and by reducing Python overhead in the tight loop. I also build a real `id->path` index once per split via `os.scandir()` to avoid repeated path construction and filesystem checks, and use a thread pool to overlap I/O + NumPy compute (safe here because `np.load`/NumPy release the GIL). All changes preserve the same evaluation semantics and should only introduce negligible floating-point differences.'
- What this solution (achieved 0.50274) has done: 'Your current 0.502 AUC strongly suggests the model is close to random, and in this competition a common cause is misaligned train/test IDs due to using a mismatched `sample_submission.csv` (6000 ids) with a different `test/` folder (often ~36k ids). I keep your exact feature extraction and LogisticRegression pipeline unchanged, but rebuild `test_ids` directly from the actual `.npy` files present under `test/` and then align/pad back to `sample_submission` ids (so the submission is always valid). This should move the score upward toward your 0.7627 target by ensuring predictions correspond to the correct test examples, while remaining minimal and stable. I also fix the (currently broken/undefined-behavior) “nonlocal_missing” increment in the feature builder, without changing computed features.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_ROOTS = [
    "/kaggle/input/seti-breakthrough-listen",
    "/kaggle/data/seti-breakthrough-listen",
    "/kaggle/input",
    "/kaggle/data",
]


def find_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


SAMPLE_PATHS = [
    "/kaggle/input/seti-breakthrough-listen/sample_submission.csv",
    "/kaggle/data/seti-breakthrough-listen/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
    "../input/sample_submission.csv",
    "../data/sample_submission.csv",
]
sample_path = find_first_existing(SAMPLE_PATHS)
if sample_path is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv in expected locations. "
        f"Tried: {SAMPLE_PATHS}"
    )

sample_sub = pd.read_csv(sample_path)
if not {"id", "target"}.issubset(sample_sub.columns):
    raise ValueError(
        f"sample_submission.csv must have columns ['id','target'], got {list(sample_sub.columns)}"
    )

TRAIN_LABEL_PATHS = [
    "/kaggle/input/seti-breakthrough-listen/train_labels.csv",
    "/kaggle/data/seti-breakthrough-listen/train_labels.csv",
    "/kaggle/input/train_labels.csv",
    "/kaggle/data/train_labels.csv",
]
train_labels_path = find_first_existing(TRAIN_LABEL_PATHS)
if train_labels_path is None:
    raise FileNotFoundError(
        "Could not find train_labels.csv in expected locations. "
        f"Tried: {TRAIN_LABEL_PATHS}"
    )
train_labels = pd.read_csv(train_labels_path)
if not {"id", "target"}.issubset(train_labels.columns):
    raise ValueError(
        f"train_labels.csv must have columns ['id','target'], got {list(train_labels.columns)}"
    )


def resolve_dataset_root():
    for root in [
        "/kaggle/input/seti-breakthrough-listen",
        "/kaggle/data/seti-breakthrough-listen",
        "/kaggle/input",
        "/kaggle/data",
        "../input",
        "../data",
    ]:
        train_dir = os.path.join(root, "train")
        test_dir = os.path.join(root, "test")
        if os.path.isdir(train_dir) and os.path.isdir(test_dir):
            return root
    raise FileNotFoundError(
        "Could not find dataset root containing both train/ and test/ folders."
    )


dataset_root = resolve_dataset_root()
train_dir = os.path.join(dataset_root, "train")
test_dir = os.path.join(dataset_root, "test")

print("Using dataset_root:", dataset_root)
print("sample_submission:", sample_path, "rows:", len(sample_sub))
print("train_labels:", train_labels_path, "rows:", len(train_labels))




## === cell 1
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import numpy as np


def id_to_path(base_dir: str, id_str: str) -> str:
    sub = id_str[0]
    return os.path.join(base_dir, sub, f"{id_str}.npy")


def build_id_to_path_index(base_dir: str):
    id2path = {}
    try:
        for subent in os.scandir(base_dir):
            if not subent.is_dir():
                continue
            subdir = subent.path
            for fent in os.scandir(subdir):
                if fent.is_file() and fent.name.endswith(".npy"):
                    id2path[fent.name[:-4]] = fent.path
    except FileNotFoundError:
        pass
    return id2path


def _quantiles_linear_from_flat_three(xf: np.ndarray):
    n = xf.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    qs = (0.10, 0.50, 0.90)
    hs = [(n - 1) * q for q in qs]
    is_ = [int(h) for h in hs]
    gs = [h - i for h, i in zip(hs, is_)]

    idxs = []
    for i, g in zip(is_, gs):
        idxs.append(i)
        if g != 0.0:
            idxs.append(i + 1)
    idxs = np.unique(np.array(idxs, dtype=np.int64))

    part = np.partition(xf, idxs)

    def qval(i, g):
        if g == 0.0:
            return float(part[i])
        xi = part[i]
        xip1 = part[i + 1]
        return float(xi + g * (xip1 - xi))

    p10 = qval(is_[0], gs[0])
    p50 = qval(is_[1], gs[1])
    p90 = qval(is_[2], gs[2])
    return p10, p50, p90


def extract_features_from_array(x: np.ndarray) -> np.ndarray:
    """
    x shape: (6, 273, 256), float16/float32.
    Keep core logic: statistical summaries + A/B differences.
    """
    x = np.asarray(x, dtype=np.float32, order="C")

    A = x[[0, 2, 4]]
    B = x[[1, 3, 5]]

    xf = x.reshape(-1)

    p10, x_median, p90 = _quantiles_linear_from_flat_three(xf)

    x_mean = float(x.mean())
    x_std = float(x.std())
    x_abs_mean = float(np.abs(x).mean())
    x_max = float(x.max())

    A_mean = float(A.mean())
    B_mean = float(B.mean())
    A_std = float(A.std())
    B_std = float(B.std())
    A_abs_mean = float(np.abs(A).mean())
    B_abs_mean = float(np.abs(B).mean())
    A_max = float(A.max())
    B_max = float(B.max())

    x_std_axis1_mean = float(x.std(axis=1).mean())
    A_std_axis1_mean = float(A.std(axis=1).mean())
    B_std_axis1_mean = float(B.std(axis=1).mean())

    feats = np.array(
        [
            x_mean,
            x_std,
            float(x_median),
            float(p90),
            float(p10),
            A_mean,
            B_mean,
            A_mean - B_mean,
            A_std,
            B_std,
            A_std - B_std,
            x_abs_mean,
            A_abs_mean - B_abs_mean,
            x_max,
            A_max - B_max,
            x_std_axis1_mean,
            (A_std_axis1_mean - B_std_axis1_mean),
        ],
        dtype=np.float32,
    )
    return feats


N_FEATS = 17


def build_feature_matrix(
    ids, base_dir: str, id2path: dict, max_workers: int = None, chunksize: int = 256
):
    n = len(ids)
    X = np.empty((n, N_FEATS), dtype=np.float32)
    missing = 0

    _load = np.load
    _extract = extract_features_from_array
    _get = id2path.get
    _id_to_path = id_to_path

    def _process_one(i_id):
        i, id_str = i_id
        p = _get(id_str)
        if p is None:
            p = _id_to_path(base_dir, id_str)
        try:
            arr = _load(p, mmap_mode="r")
        except FileNotFoundError:
            return i, None
        return i, _extract(arr)

    if max_workers is None:
        cpu = os.cpu_count() or 4
        max_workers = min(12, max(4, cpu))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        it = enumerate(ids)
        while True:
            batch = []
            for _ in range(chunksize):
                try:
                    batch.append(next(it))
                except StopIteration:
                    break
            if not batch:
                break
            futures.extend(ex.submit(_process_one, item) for item in batch)

        for fut in as_completed(futures):
            i, feats = fut.result()
            if feats is None:
                missing += 1
                X[i] = 0.0
            else:
                X[i] = feats

    if missing:
        print(f"Warning: missing {missing} files under {base_dir}")
    return X


def list_ids_from_npy_tree(base_dir: str):
    ids = []
    for subent in os.scandir(base_dir):
        if not subent.is_dir():
            continue
        for fent in os.scandir(subent.path):
            if fent.is_file() and fent.name.endswith(".npy"):
                ids.append(fent.name[:-4])
    ids.sort()
    return ids


train_id2path = build_id_to_path_index(train_dir)

train_ids = train_labels["id"].astype(str).tolist()
y = train_labels["target"].astype(int).values
X_train = build_feature_matrix(train_ids, train_dir, train_id2path)

print(
    "X_train shape:", X_train.shape, "y shape:", y.shape, "pos_rate:", float(y.mean())
)




## === cell 2
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

clf = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        (
            "lr",
            LogisticRegression(
                solver="lbfgs",
                max_iter=300,
                C=1.0,
                random_state=RANDOM_STATE,
                n_jobs=None,
            ),
        ),
    ]
)

clf.fit(X_train, y)

test_id2path = build_id_to_path_index(test_dir)

test_ids_actual = list_ids_from_npy_tree(test_dir)
print(
    "Detected test files:",
    len(test_ids_actual),
    "sample_submission rows:",
    len(sample_sub),
)

X_test_actual = build_feature_matrix(test_ids_actual, test_dir, test_id2path)

pred_actual = clf.predict_proba(X_test_actual)[:, 1].astype(np.float64)
pred_actual = np.clip(pred_actual, 0.0, 1.0)

pred_map = dict(zip(test_ids_actual, pred_actual))

test_ids_submit = sample_sub["id"].astype(str).tolist()
pred_submit = np.array(
    [pred_map.get(i, 0.5) for i in test_ids_submit], dtype=np.float64
)

data1 = pd.DataFrame({"id": test_ids_submit, "target": pred_submit})

print("Pred summary:", pd.Series(pred_submit).describe())




## === cell 3
data11 = data1.copy()
data11["target"] = data11["target"].clip(0.0, 1.0)

data11 = data11[["id", "target"]]
if data11["id"].isna().any():
    raise ValueError("Found missing ids in the final submission.")
if len(data11) != len(sample_sub):
    raise ValueError(
        f"Submission row count {len(data11)} != sample_submission row count {len(sample_sub)}"
    )

out_path = "submission.csv"
data11.to_csv(out_path, index=False)
print(f"Wrote {out_path} with shape {data11.shape} and columns {list(data11.columns)}")
print(data11.head())
