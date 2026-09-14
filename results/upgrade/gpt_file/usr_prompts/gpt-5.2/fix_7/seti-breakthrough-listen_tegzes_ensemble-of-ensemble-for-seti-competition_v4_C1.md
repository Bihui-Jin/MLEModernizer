# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42

BASE_CANDIDATES = [
    "/kaggle/input/seti-breakthrough-listen",
    "/kaggle/data/seti-breakthrough-listen",
    "/kaggle/input",
    "/kaggle/data",
]
BASE = None
for b in BASE_CANDIDATES:
    if os.path.exists(b):
        if os.path.exists(os.path.join(b, "train_labels.csv")) or os.path.exists(
            os.path.join(b, "seti-breakthrough-listen", "train_labels.csv")
        ):
            BASE = b
            break
if BASE is None:
    BASE = "/kaggle/input/seti-breakthrough-listen"


def resolve_path(*parts):
    p = os.path.join(BASE, *parts)
    if os.path.exists(p):
        return p
    p2 = os.path.join(BASE, "seti-breakthrough-listen", *parts)
    if os.path.exists(p2):
        return p2
    p3 = os.path.join("/kaggle/data/seti-breakthrough-listen", *parts)
    if os.path.exists(p3):
        return p3
    return p  # fall back (will error later if truly missing)


TRAIN_LABELS_PATH = resolve_path("train_labels.csv")
TRAIN_DIR = resolve_path("train")
TEST_DIR = resolve_path("test")
SAMPLE_SUB_PATH = resolve_path("sample_submission.csv")

OLD_TRAIN_LABELS_PATH = resolve_path("old_leaky_data", "train_labels_old.csv")
OLD_TEST_LABELS_PATH = resolve_path("old_leaky_data", "test_labels_old.csv")
OLD_TRAIN_DIR = resolve_path("old_leaky_data", "train_old")
OLD_TEST_DIR = resolve_path("old_leaky_data", "test_old")

print("Using paths:")
print("TRAIN_LABELS_PATH:", TRAIN_LABELS_PATH)
print("TRAIN_DIR:", TRAIN_DIR)
print("TEST_DIR:", TEST_DIR)
print("SAMPLE_SUB_PATH:", SAMPLE_SUB_PATH)
print(
    "OLD_TRAIN_LABELS_PATH:",
    OLD_TRAIN_LABELS_PATH,
    "exists:",
    os.path.exists(OLD_TRAIN_LABELS_PATH),
)
print(
    "OLD_TEST_LABELS_PATH:",
    OLD_TEST_LABELS_PATH,
    "exists:",
    os.path.exists(OLD_TEST_LABELS_PATH),
)
print("OLD_TRAIN_DIR:", OLD_TRAIN_DIR, "exists:", os.path.exists(OLD_TRAIN_DIR))
print("OLD_TEST_DIR:", OLD_TEST_DIR, "exists:", os.path.exists(OLD_TEST_DIR))



## === cell 1
train_labels = pd.read_csv(TRAIN_LABELS_PATH)
assert {"id", "target"}.issubset(train_labels.columns)
train_labels["id"] = train_labels["id"].astype(str)
train_labels["target"] = train_labels["target"].astype(int)

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
assert {"id", "target"}.issubset(sample_sub.columns)
sample_sub["id"] = sample_sub["id"].astype(str)

print("train_labels:", train_labels.shape)
print("sample_sub:", sample_sub.shape)

old_train_labels = None
old_test_labels = None
if os.path.exists(OLD_TRAIN_LABELS_PATH) and os.path.exists(OLD_TEST_LABELS_PATH):
    old_train_labels = pd.read_csv(OLD_TRAIN_LABELS_PATH)
    old_test_labels = pd.read_csv(OLD_TEST_LABELS_PATH)
    for df in (old_train_labels, old_test_labels):
        assert {"id", "target"}.issubset(df.columns)
        df["id"] = df["id"].astype(str)
        df["target"] = df["target"].astype(int)
    print("old_train_labels:", old_train_labels.shape)
    print("old_test_labels:", old_test_labels.shape)
else:
    print("Old labeled data not found; proceeding with main training labels only.")




## === cell 2
def id_to_path(root_dir: str, fid: str) -> str:
    return os.path.join(root_dir, fid[0], f"{fid}.npy")


def _index_npy_files(root_dir: str) -> dict:
    out = {}
    if not os.path.exists(root_dir):
        return out
    for sub in os.listdir(root_dir):
        subdir = os.path.join(root_dir, sub)
        if not os.path.isdir(subdir):
            continue
        try:
            for fn in os.listdir(subdir):
                if fn.endswith(".npy"):
                    fid = fn[:-4]
                    out[fid] = os.path.join(subdir, fn)
        except FileNotFoundError:
            continue
    return out


def build_file_map_from_ids(root_dir: str, ids) -> dict:
    idx = _index_npy_files(root_dir)
    return {fid: idx[fid] for fid in ids if fid in idx}


train_ids_all = train_labels["id"].tolist()
test_ids_all = sample_sub["id"].tolist()

train_files = build_file_map_from_ids(TRAIN_DIR, train_ids_all)
test_files = build_file_map_from_ids(TEST_DIR, test_ids_all)

print("Found train npy files:", len(train_files))
print("Found test npy files:", len(test_files))

train_df = train_labels[train_labels["id"].isin(train_files.keys())].copy()
print("Train rows with existing files:", train_df.shape)

missing_test = [i for i in test_ids_all if i not in test_files]
print("Missing test files (should be 0):", len(missing_test))

old_train_files = {}
old_test_files = {}
old_df = None
if (
    old_train_labels is not None
    and old_test_labels is not None
    and os.path.exists(OLD_TRAIN_DIR)
    and os.path.exists(OLD_TEST_DIR)
):
    old_train_files = build_file_map_from_ids(
        OLD_TRAIN_DIR, old_train_labels["id"].tolist()
    )
    old_test_files = build_file_map_from_ids(
        OLD_TEST_DIR, old_test_labels["id"].tolist()
    )

    old_train_df = old_train_labels[
        old_train_labels["id"].isin(old_train_files.keys())
    ].copy()
    old_test_df = old_test_labels[
        old_test_labels["id"].isin(old_test_files.keys())
    ].copy()
    old_df = pd.concat([old_train_df, old_test_df], axis=0, ignore_index=True)

    print(
        "Found old_train npy files:", len(old_train_files), "rows:", old_train_df.shape
    )
    print("Found old_test npy files:", len(old_test_files), "rows:", old_test_df.shape)
    print("Old labeled rows with existing files (combined):", old_df.shape)
else:
    print("Old labeled data not usable; skipping.")




## === cell 3
def extract_features_from_array(x):
    x = x.astype(np.float32, copy=False)
    A = x[[0, 2, 4]]
    O = x[[1, 3, 5]]

    A_mean = A.mean(axis=(1, 2))  # (3,)
    O_mean = O.mean(axis=(1, 2))
    A_std = A.std(axis=(1, 2))
    O_std = O.std(axis=(1, 2))

    A_mean_m = A_mean.mean()
    O_mean_m = O_mean.mean()
    A_std_m = A_std.mean()
    O_std_m = O_std.mean()

    A_max = np.max(A)
    O_max = np.max(O)

    A_p995 = np.percentile(A, [99.5])[0]
    O_p995 = np.percentile(O, [99.5])[0]

    A_t = A.mean(axis=2)  # (3, H)
    O_t = O.mean(axis=2)
    A_f = A.mean(axis=1)  # (3, W)
    O_f = O.mean(axis=1)

    A_t_std_m = A_t.std(axis=1).mean()
    O_t_std_m = O_t.std(axis=1).mean()
    A_f_std_m = A_f.std(axis=1).mean()
    O_f_std_m = O_f.std(axis=1).mean()

    A_sq_m = np.mean(A * A)
    O_sq_m = np.mean(O * O)

    f = [
        A_mean_m,
        O_mean_m,
        A_std_m,
        O_std_m,
        (A_mean_m - O_mean_m),
        (A_std_m - O_std_m),
        A_max,
        O_max,
        (A_max - O_max),
        A_p995,
        O_p995,
        (A_p995 - O_p995),
        A_t_std_m,
        O_t_std_m,
        (A_t_std_m - O_t_std_m),
        A_f_std_m,
        O_f_std_m,
        (A_f_std_m - O_f_std_m),
        A_sq_m,
        O_sq_m,
        (A_sq_m - O_sq_m),
    ]
    return np.asarray(f, dtype=np.float32)


from concurrent.futures import ThreadPoolExecutor


def featurize_ids(id_list, file_map, n_features=None, max_workers=None, chunksize=256):
    if not id_list:
        raise ValueError("Empty id_list")

    if n_features is None:
        first_id = id_list[0]
        n_features = int(
            extract_features_from_array(
                np.load(file_map[first_id], allow_pickle=False, mmap_mode="r")
            ).shape[0]
        )

    X = np.empty((len(id_list), n_features), dtype=np.float32)

    def _one(i, fid):
        arr = np.load(file_map[fid], allow_pickle=False, mmap_mode="r")
        feat = extract_features_from_array(arr)
        if feat.shape[0] != n_features:
            raise ValueError(
                f"Feature length mismatch for id={fid}: got {feat.shape[0]}, expected {n_features}"
            )
        X[i] = feat

    if max_workers is None:
        cpu = os.cpu_count() or 4
        max_workers = min(8, cpu)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for start in range(0, len(id_list), chunksize):
            end = min(len(id_list), start + chunksize)
            futures = [ex.submit(_one, i, id_list[i]) for i in range(start, end)]
            for f in futures:
                f.result()
    return X


tmp_id = train_df["id"].iloc[0]
tmp_feat = extract_features_from_array(
    np.load(train_files[tmp_id], allow_pickle=False, mmap_mode="r")
)
print("Feature dim:", tmp_feat.shape[0])



## === cell 4
if old_df is not None and len(old_df) > 0:
    combined_df = pd.concat([train_df, old_df], axis=0, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=["id"], keep="first").reset_index(
        drop=True
    )

    combined_files = dict(train_files)
    combined_files.update(old_train_files)
    combined_files.update(old_test_files)

    print("Combined labeled rows (after dedup):", combined_df.shape)
else:
    combined_df = train_df
    combined_files = train_files
    print("Combined labeled rows:", combined_df.shape)

train_ids = combined_df["id"].tolist()
y = combined_df["target"].values.astype(np.int32)

X = featurize_ids(train_ids, combined_files, n_features=None)

print("X:", X.shape, "y:", y.shape, "pos_rate:", float(y.mean()))



## === cell 5
model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, solver="lbfgs", n_jobs=None)),
    ]
)

model.fit(X, y)

try:
    from sklearn.metrics import roc_auc_score

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    va_pred = model.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, va_pred)
    print("Validation AUC (sanity check only; model trained on full data):", float(auc))
except Exception as e:
    print("AUC check skipped due to:", repr(e))



## === cell 6
test_ids_ordered = sample_sub["id"].tolist()

n_features = int(X.shape[1])
X_test = np.empty((len(test_ids_ordered), n_features), dtype=np.float32)
missing_mask = np.zeros(len(test_ids_ordered), dtype=bool)

present_idx = []
for i, fid in enumerate(test_ids_ordered):
    if fid not in test_files:
        missing_mask[i] = True
    else:
        present_idx.append(i)


def _fill_one(i):
    fid = test_ids_ordered[i]
    arr = np.load(test_files[fid], allow_pickle=False, mmap_mode="r")
    feat = extract_features_from_array(arr)
    if feat.shape[0] != n_features:
        raise ValueError(
            f"Test feature length mismatch for id={fid}: got {feat.shape[0]}, expected {n_features}"
        )
    X_test[i] = feat


cpu = os.cpu_count() or 4
max_workers = min(8, cpu)
chunksize = 512
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for start in range(0, len(present_idx), chunksize):
        end = min(len(present_idx), start + chunksize)
        futures = [ex.submit(_fill_one, present_idx[j]) for j in range(start, end)]
        for f in futures:
            f.result()

test_pred = model.predict_proba(X_test)[:, 1]
if missing_mask.any():
    test_pred[missing_mask] = 0.5

sub = pd.DataFrame({"id": test_ids_ordered, "target": test_pred.astype(np.float64)})
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("target min/max:", float(sub["target"].min()), float(sub["target"].max()))
