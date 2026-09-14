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
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_CPU = os.cpu_count() or 1
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

np.random.seed(42)



## === cell 1
_CANDIDATE_BASES = [
    "/kaggle/input",  # common Kaggle notebooks
    "/kaggle/data",  # this environment's provided tree shows data here
    "/kaggle/input/seti-breakthrough-listen",  # sometimes competition nested like this
    "/kaggle/data/seti-breakthrough-listen",
]


def _pick_base(candidates):
    for b in candidates:
        if os.path.isdir(b):
            train_dir = os.path.join(b, "train")
            test_dir = os.path.join(b, "test")
            labels = os.path.join(b, "train_labels.csv")
            sample = os.path.join(b, "sample_submission.csv")
            if (
                os.path.isdir(train_dir)
                and os.path.isdir(test_dir)
                and os.path.exists(labels)
                and os.path.exists(sample)
            ):
                return b
    return "/kaggle/input"


BASE = _pick_base(_CANDIDATE_BASES)
TRAIN_DIR = os.path.join(BASE, "train")
TEST_DIR = os.path.join(BASE, "test")
TRAIN_LABELS_PATH = os.path.join(BASE, "train_labels.csv")
SAMPLE_SUB_PATH = os.path.join(BASE, "sample_submission.csv")

assert os.path.exists(TRAIN_LABELS_PATH), f"Missing: {TRAIN_LABELS_PATH} (BASE={BASE})"
assert os.path.exists(SAMPLE_SUB_PATH), f"Missing: {SAMPLE_SUB_PATH} (BASE={BASE})"
assert os.path.isdir(TRAIN_DIR), f"Missing dir: {TRAIN_DIR} (BASE={BASE})"
assert os.path.isdir(TEST_DIR), f"Missing dir: {TEST_DIR} (BASE={BASE})"

train_labels = pd.read_csv(TRAIN_LABELS_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

if list(sample_sub.columns) != ["id", "target"]:
    sample_sub = sample_sub.rename(
        columns={sample_sub.columns[0]: "id", sample_sub.columns[1]: "target"}
    )
train_labels = train_labels[["id", "target"]].copy()

print("Using BASE:", BASE)
print("Train labels:", train_labels.shape, "Sample sub:", sample_sub.shape)
train_labels.head(), sample_sub.head()



## === cell 2
A_IDX = [0, 2, 4]
OFF_IDX = [1, 3, 5]


def build_id_to_path(root_dir: str) -> dict:
    id2path = {}
    with os.scandir(root_dir) as it:
        for subent in it:
            if not subent.is_dir():
                continue
            subdir = subent.path
            with os.scandir(subdir) as it2:
                for ent in it2:
                    if ent.is_file() and ent.name.endswith(".npy"):
                        _id = ent.name[:-4]
                        id2path[_id] = ent.path
    return id2path


TRAIN_ID2PATH = build_id_to_path(TRAIN_DIR)
TEST_ID2PATH = build_id_to_path(TEST_DIR)


def _safe_load_npy(path: str) -> np.ndarray:
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return arr


def extract_features_both_from_arr(x: np.ndarray):
    A = x[A_IDX]  # (3, 273, 256)
    O = x[OFF_IDX]  # (3, 273, 256)

    A_mean = A.mean(axis=(1, 2))
    A_std = A.std(axis=(1, 2))
    O_mean = O.mean(axis=(1, 2))
    O_std = O.std(axis=(1, 2))

    A_mean_mean = A_mean.mean()
    A_mean_std = A_mean.std()
    O_mean_mean = O_mean.mean()
    O_mean_std = O_mean.std()

    A_std_mean = A_std.mean()
    A_std_std = A_std.std()
    O_std_mean = O_std.mean()
    O_std_std = O_std.std()

    d_mean = A_mean_mean - O_mean_mean
    d_std = A_std_mean - O_std_mean

    A_absmean = np.abs(A).mean()
    O_absmean = np.abs(O).mean()
    A_max = A.max()
    O_max = O.max()

    A_flat = A.reshape(-1)
    O_flat = O.reshape(-1)

    A_q95, A_q99 = np.percentile(A_flat, [95.0, 99.0])
    O_q95, O_q99 = np.percentile(O_flat, [95.0, 99.0])

    A_time = A.mean(axis=2)  # (3, 273)
    O_time = O.mean(axis=2)  # (3, 273)
    A_freq = A.mean(axis=1)  # (3, 256)
    O_freq = O.mean(axis=1)  # (3, 256)

    A_time_var = A_time.var()
    O_time_var = O_time.var()
    A_freq_var = A_freq.var()
    O_freq_var = O_freq.var()

    D = A.mean(axis=0) - O.mean(axis=0)  # (273, 256)
    D_absmean = np.abs(D).mean()
    D_max = D.max()
    D_q95 = np.percentile(D.reshape(-1), 95.0)
    D_time_var = D.mean(axis=1).var()
    D_freq_var = D.mean(axis=0).var()

    DA = (A - O).astype(np.float32, copy=False)  # (3,273,256)
    DA_abs = np.abs(DA)
    DA_mean = DA.mean(axis=(1, 2))  # per A-slot difference mean
    DA_std = DA.std(axis=(1, 2))  # per A-slot difference std

    DA_absmean = DA_abs.mean()
    DA_absmax = DA_abs.max()

    DA_q95, DA_q99 = np.percentile(DA.reshape(-1), [95.0, 99.0])
    DA_absq95, DA_absq99 = np.percentile(DA_abs.reshape(-1), [95.0, 99.0])

    DA_mean_std = DA_mean.std()
    DA_std_std = DA_std.std()

    base_feats = np.array(
        [
            A_mean_mean,
            A_mean_std,
            O_mean_mean,
            O_mean_std,
            A_std_mean,
            A_std_std,
            O_std_mean,
            O_std_std,
            d_mean,
            d_std,
            A_absmean,
            O_absmean,
            A_absmean - O_absmean,
            A_max,
            O_max,
            A_max - O_max,
            A_q95,
            O_q95,
            A_q95 - O_q95,
            A_q99,
            O_q99,
            A_q99 - O_q99,
            A_time_var,
            O_time_var,
            A_time_var - O_time_var,
            A_freq_var,
            O_freq_var,
            A_freq_var - O_freq_var,
            D_absmean,
            D_max,
            D_q95,
            D_time_var,
            D_freq_var,
            DA_mean.mean(),
            DA_mean_std,
            DA_std.mean(),
            DA_std_std,
            DA_absmean,
            DA_absmax,
            DA_q95,
            DA_q99,
            DA_absq95,
            DA_absq99,
        ],
        dtype=np.float32,
    )

    eps = 1e-8
    A_time_profiles = A_time  # (3,273)
    O_time_profiles = O_time
    A_freq_profiles = A_freq  # (3,256)
    O_freq_profiles = O_freq

    def _cos(u, v):
        uu = float(np.dot(u, u))
        vv = float(np.dot(v, v))
        if uu <= eps or vv <= eps:
            return 0.0
        return float(np.dot(u, v) / (np.sqrt(uu * vv) + eps))

    A_t01 = _cos(A_time_profiles[0], A_time_profiles[1])
    A_t12 = _cos(A_time_profiles[1], A_time_profiles[2])
    A_t02 = _cos(A_time_profiles[0], A_time_profiles[2])
    O_t01 = _cos(O_time_profiles[0], O_time_profiles[1])
    O_t12 = _cos(O_time_profiles[1], O_time_profiles[2])
    O_t02 = _cos(O_time_profiles[0], O_time_profiles[2])

    A_f01 = _cos(A_freq_profiles[0], A_freq_profiles[1])
    A_f12 = _cos(A_freq_profiles[1], A_freq_profiles[2])
    A_f02 = _cos(A_freq_profiles[0], A_freq_profiles[2])
    O_f01 = _cos(O_freq_profiles[0], O_freq_profiles[1])
    O_f12 = _cos(O_freq_profiles[1], O_freq_profiles[2])
    O_f02 = _cos(O_freq_profiles[0], O_freq_profiles[2])

    A_time_mean_prof = A_time_profiles.mean(axis=0)
    O_time_mean_prof = O_time_profiles.mean(axis=0)
    A_freq_mean_prof = A_freq_profiles.mean(axis=0)
    O_freq_mean_prof = O_freq_profiles.mean(axis=0)

    AO_time_cos = _cos(A_time_mean_prof, O_time_mean_prof)
    AO_freq_cos = _cos(A_freq_mean_prof, O_freq_mean_prof)

    D_flat = D.reshape(-1)
    absD = np.abs(D_flat)
    k = 128
    if absD.size >= k:
        topk = np.partition(absD, -k)[-k:]
        D_topk_mean = float(topk.mean())
        D_topk_sum_frac = float(topk.sum() / (absD.sum() + eps))
    else:
        D_topk_mean = float(absD.mean())
        D_topk_sum_frac = 1.0

    ext_feats = np.array(
        [
            (A_t01 + A_t12 + A_t02) / 3.0,
            (A_f01 + A_f12 + A_f02) / 3.0,
            (O_t01 + O_t12 + O_t02) / 3.0,
            (O_f01 + O_f12 + O_f02) / 3.0,
            AO_time_cos,
            AO_freq_cos,
            D_topk_mean,
            D_topk_sum_frac,
        ],
        dtype=np.float32,
    )

    return base_feats, np.concatenate([base_feats, ext_feats], axis=0)


def extract_features_from_arr(x: np.ndarray, mode: str = "extended") -> np.ndarray:
    b, e = extract_features_both_from_arr(x)
    return b if mode == "base" else e


from concurrent.futures import ThreadPoolExecutor


def _feat_worker_both(idx_id_target):
    idx, _id, yy, id2path = idx_id_target
    path = id2path.get(_id, "")
    if not path:
        return idx, None, None, yy
    arr = _safe_load_npy(path)
    b, e = extract_features_both_from_arr(arr)
    return idx, b, e, yy




## === cell 3
RANDOM_STATE = 42
MAX_TRAIN_SAMPLES = 54000  # use full train_labels.csv by default

train_labels_avail = train_labels[
    train_labels["id"].astype(str).isin(TRAIN_ID2PATH.keys())
].copy()
train_labels_avail = train_labels_avail.reset_index(drop=True)

if len(train_labels_avail) == 0:
    raise RuntimeError(
        f"No train ids found on disk under {TRAIN_DIR}; check dataset structure/base path."
    )

train_df = train_labels_avail
if MAX_TRAIN_SAMPLES is not None and MAX_TRAIN_SAMPLES < len(train_df):
    train_df = train_df.sample(
        n=int(MAX_TRAIN_SAMPLES),
        random_state=RANDOM_STATE,
        replace=False,
    ).reset_index(drop=True)

FEAT_DIM_BASE = int(
    extract_features_from_arr(
        np.zeros((6, 273, 256), dtype=np.float32), mode="base"
    ).shape[0]
)
FEAT_DIM_EXT = int(
    extract_features_from_arr(
        np.zeros((6, 273, 256), dtype=np.float32), mode="extended"
    ).shape[0]
)

items_both = [
    (i, _id, int(yy), TRAIN_ID2PATH)
    for i, (_id, yy) in enumerate(
        zip(train_df["id"].astype(str).values, train_df["target"].values)
    )
]


def _load_features_both(items, feat_dim_base, feat_dim_ext):
    Xb = np.empty((len(items), feat_dim_base), dtype=np.float32)
    Xe = np.empty((len(items), feat_dim_ext), dtype=np.float32)
    y_ = np.empty((len(items),), dtype=np.int32)
    n_loaded_ = 0
    missing_ = 0

    max_workers_ = min(32, _CPU)
    with ThreadPoolExecutor(max_workers=max_workers_) as ex:
        for idx, fb, fe, yy in ex.map(_feat_worker_both, items, chunksize=128):
            if fb is None:
                missing_ += 1
                continue
            Xb[n_loaded_] = fb
            Xe[n_loaded_] = fe
            y_[n_loaded_] = yy
            n_loaded_ += 1

    return Xb[:n_loaded_], Xe[:n_loaded_], y_[:n_loaded_], missing_, max_workers_


X_base, X_ext, y, missing_train, max_workers = _load_features_both(
    items_both, FEAT_DIM_BASE, FEAT_DIM_EXT
)

print(
    f"Train candidates: {len(train_labels)}, available on disk: {len(train_labels_avail)}, used: {len(train_df)}\n"
    f"Loaded: {len(y)} (missing {missing_train}), X_base shape: {X_base.shape}, X_ext shape: {X_ext.shape}"
)

assert len(y) > 1000, "Too few training samples loaded; check paths/dataset structure."

X = None
FEAT_DIM = None
FEATURE_MODE_SELECTED = None



## === cell 4
RANDOM_STATE = 42
try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None

C_CANDIDATES = [0.05, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]


def _infer_group_from_path(id2path: dict, ids: np.ndarray) -> np.ndarray:
    groups = np.full(len(ids), -1, dtype=np.int32)
    for i, _id in enumerate(ids):
        p = id2path.get(str(_id), "")
        if p:
            groups[i] = int(os.path.basename(os.path.dirname(p)))
    groups[groups < 0] = 0
    return groups


def _group_holdout_split(yvec: np.ndarray, groups: np.ndarray, test_size: float = 0.2):
    rng = np.random.RandomState(RANDOM_STATE)
    uniq = np.unique(groups)

    pos_rate = np.array([yvec[groups == g].mean() for g in uniq], dtype=np.float64)
    bins = np.quantile(pos_rate, [0.0, 0.25, 0.5, 0.75, 1.0])
    bins = np.unique(bins)
    if len(bins) < 3:
        rng.shuffle(uniq)
        n_va = max(1, int(round(len(uniq) * test_size)))
        va_g = set(uniq[:n_va].tolist())
    else:
        bin_id = np.digitize(pos_rate, bins[1:-1], right=True)
        va_g = set()
        for b in np.unique(bin_id):
            g_in = uniq[bin_id == b]
            rng.shuffle(g_in)
            n_va = max(1, int(round(len(g_in) * test_size)))
            va_g.update(g_in[:n_va].tolist())

    va_mask = np.isin(groups, list(va_g))
    tr_idx = np.where(~va_mask)[0]
    va_idx = np.where(va_mask)[0]
    return tr_idx, va_idx


def _fit_select_C(Xmat, yvec, groups=None):
    if groups is None:
        X_tr, X_va, y_tr, y_va = train_test_split(
            Xmat, yvec, test_size=0.2, random_state=RANDOM_STATE, stratify=yvec
        )
    else:
        tr_idx, va_idx = _group_holdout_split(yvec, groups, test_size=0.2)
        X_tr, X_va, y_tr, y_va = Xmat[tr_idx], Xmat[va_idx], yvec[tr_idx], yvec[va_idx]

    best_auc = -1.0
    best_model = None
    best_C = None
    best_fold_model = None

    for C in C_CANDIDATES:
        lr_kwargs = dict(
            max_iter=1200,
            solver="lbfgs",
            C=float(C),
            random_state=RANDOM_STATE,
            class_weight=None,
        )
        try:
            lr_kwargs["n_jobs"] = _CPU
        except Exception:
            pass

        m = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("clf", LogisticRegression(**lr_kwargs)),
            ]
        )
        m.fit(X_tr, y_tr)

        if roc_auc_score is not None:
            va_pred = m.predict_proba(X_va)[:, 1]
            auc = roc_auc_score(y_va, va_pred)
            print(f"  Holdout AUC (C={C}): {auc:.6f}")
            if auc > best_auc:
                best_auc = auc
                best_model = m
                best_fold_model = m  # trained only on X_tr
                best_C = C
        else:
            best_model = m
            best_fold_model = m
            best_C = C

    return best_model, best_fold_model, best_C, best_auc, (X_tr, X_va, y_tr, y_va)


print("Selecting feature mode using holdout AUC...")

train_ids_order = train_df["id"].astype(str).values
loaded_mask = np.array([_id in TRAIN_ID2PATH for _id in train_ids_order], dtype=bool)
train_ids_loaded = train_ids_order[loaded_mask][: len(y)]
groups_loaded = _infer_group_from_path(TRAIN_ID2PATH, train_ids_loaded)

print("Mode=base")
model_base, fold_base, C_base, auc_base, split_base = _fit_select_C(
    X_base, y, groups=groups_loaded
)
print("Mode=extended")
model_ext, fold_ext, C_ext, auc_ext, split_ext = _fit_select_C(
    X_ext, y, groups=groups_loaded
)

if roc_auc_score is None:
    FEATURE_MODE_SELECTED = "extended"
else:
    FEATURE_MODE_SELECTED = "extended" if auc_ext > auc_base + 1e-6 else "base"

if FEATURE_MODE_SELECTED == "extended":
    X = X_ext
    FEAT_DIM = X_ext.shape[1]
    best_model = model_ext
    best_fold_model = fold_ext
    best_C = C_ext
    best_auc = auc_ext
    X_tr, X_va, y_tr, y_va = split_ext
else:
    X = X_base
    FEAT_DIM = X_base.shape[1]
    best_model = model_base
    best_fold_model = fold_base
    best_C = C_base
    best_auc = auc_base
    X_tr, X_va, y_tr, y_va = split_base

print("Selected feature mode:", FEATURE_MODE_SELECTED)
print("Selected C:", best_C, "Best holdout AUC:", best_auc if best_auc >= 0 else "N/A")

model = best_model
model.fit(X, y)

calibrator = None
use_calibration = False
if roc_auc_score is not None and best_fold_model is not None:
    va_proba_raw = best_fold_model.predict_proba(X_va)[:, 1].astype(np.float64)
    auc_raw = roc_auc_score(y_va, va_proba_raw)

    va_proba = np.clip(va_proba_raw, 1e-6, 1 - 1e-6)
    va_logit = np.log(va_proba / (1.0 - va_proba)).reshape(-1, 1)

    calibrator = LogisticRegression(
        solver="lbfgs",
        C=1.0,
        max_iter=500,
        random_state=RANDOM_STATE,
        class_weight=None,
    )
    calibrator.fit(va_logit, y_va)
    cal_pred = calibrator.predict_proba(va_logit)[:, 1]
    auc_cal = roc_auc_score(y_va, cal_pred)

    use_calibration = bool(auc_cal > auc_raw + 1e-6)
    print(
        f"Holdout AUC raw: {auc_raw:.6f} | after Platt: {auc_cal:.6f} | use_calibration={use_calibration}"
    )

    if not use_calibration:
        calibrator = None



## === cell 5
test_ids = sample_sub["id"].astype(str).values

preds = np.full(len(test_ids), 0.5, dtype=np.float32)
missing_test = 0

X_test = np.empty((len(test_ids), FEAT_DIM), dtype=np.float32)
have = np.zeros(len(test_ids), dtype=bool)


def _feat_worker_single(idx_id_target):
    idx, _id, id2path, feat_mode = idx_id_target
    path = id2path.get(_id, "")
    if not path:
        return idx, None
    arr = _safe_load_npy(path)
    fb, fe = extract_features_both_from_arr(arr)
    return idx, (fb if feat_mode == "base" else fe)


test_items = [
    (i, _id, TEST_ID2PATH, FEATURE_MODE_SELECTED) for i, _id in enumerate(test_ids)
]

with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for idx, feats in ex.map(_feat_worker_single, test_items, chunksize=128):
        if feats is None:
            missing_test += 1
            continue
        X_test[idx] = feats
        have[idx] = True

if have.any():
    base_pred = model.predict_proba(X_test[have])[:, 1].astype(np.float64)
    if calibrator is not None:
        base_pred = np.clip(base_pred, 1e-6, 1 - 1e-6)
        base_logit = np.log(base_pred / (1.0 - base_pred)).reshape(-1, 1)
        base_pred = calibrator.predict_proba(base_logit)[:, 1]
    preds[have] = base_pred.astype(np.float32)

print(f"Feature mode used for test: {FEATURE_MODE_SELECTED}")
print(f"Test size: {len(test_ids)}, missing test npy: {missing_test}")



## === cell 6
submission = pd.DataFrame({"id": test_ids, "target": preds.astype(np.float32)})
submission["target"] = submission["target"].clip(0.0, 1.0)

out_path = "submission.csv"
submission.to_csv(out_path, index=False)

print(submission.head())
print("Wrote:", out_path, "rows:", len(submission), "cols:", list(submission.columns))
assert os.path.exists(out_path) and out_path.endswith(".csv")
assert len(submission) == len(sample_sub)
assert list(submission.columns) == ["id", "target"]
