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

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache



## === cell 1
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
    for b in BASE_CANDIDATES:
        if os.path.exists(b):
            BASE = b
            break


def resolve_path(*parts):
    return os.path.join(BASE, *parts)


if os.path.exists(resolve_path("train_labels.csv")):
    DATA_ROOT = BASE
else:
    DATA_ROOT = resolve_path("seti-breakthrough-listen")

TRAIN_LABELS_PATH = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")

assert os.path.exists(
    TRAIN_LABELS_PATH
), f"Missing train_labels.csv at {TRAIN_LABELS_PATH}"
assert os.path.exists(
    SAMPLE_SUB_PATH
), f"Missing sample_submission.csv at {SAMPLE_SUB_PATH}"
assert os.path.isdir(TRAIN_DIR), f"Missing train/ dir at {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing test/ dir at {TEST_DIR}"

train_labels = pd.read_csv(TRAIN_LABELS_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

train_labels.head(), sample_sub.head()




## === cell 2
def id_to_npy_path(root_dir: str, fid: str) -> str:
    return os.path.join(root_dir, fid[0], f"{fid}.npy")


def build_id_to_path_map_from_ids(root_dir: str, ids):
    ids = list(ids)
    out = {}
    missing = []
    for fid in ids:
        p = id_to_npy_path(root_dir, fid)
        if os.path.exists(p):
            out[fid] = p
        else:
            missing.append(fid)
    return out, missing


train_labels = train_labels.drop_duplicates(subset=["id"], keep="first").copy()

train_map, missing_train_ids = build_id_to_path_map_from_ids(
    TRAIN_DIR, train_labels["id"].tolist()
)
train_df = train_labels[train_labels["id"].isin(train_map.keys())].reset_index(
    drop=True
)
train_ids = train_df["id"].tolist()

if len(train_ids) == 0:
    raise RuntimeError("No overlap between train_labels ids and train .npy files.")

if missing_train_ids:
    print(
        f"Warning: {len(missing_train_ids)} train ids have no corresponding .npy files; they will be skipped."
    )

test_ids_for_map = sample_sub["id"].tolist()
test_map, missing_test_ids = build_id_to_path_map_from_ids(TEST_DIR, test_ids_for_map)

print("Train rows used:", len(train_df))
print("Train files mapped:", len(train_map))
print("Test files mapped:", len(test_map))
print("Sample submission rows:", len(sample_sub))

assert (
    train_df["id"].tolist() == train_ids
), "train_df order must exactly match train_ids."
assert all(
    fid in train_map for fid in train_ids
), "Every used train id must have a file."
assert sample_sub[
    "id"
].is_unique, "sample_submission contains duplicate ids unexpectedly."
if missing_test_ids:
    raise FileNotFoundError(
        f"Missing {len(missing_test_ids)} test .npy files, e.g. {missing_test_ids[:5]}"
    )




## === cell 3
def _pct_10_90(arr):
    q10, q90 = np.quantile(arr, (0.1, 0.9), method="linear")
    return float(q10), float(q90)


def _pct_10_90_1d(arr1d):
    q10, q90 = np.quantile(arr1d, (0.1, 0.9), method="linear")
    return float(q10), float(q90)


def extract_features_from_array(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)

    A = x[[0, 2, 4]]  # (3,273,256)
    O = x[[1, 3, 5]]  # (3,273,256)

    Amean = A.mean(axis=0)
    Omean = O.mean(axis=0)
    D = Amean - Omean

    feats = []
    for arr in (Amean, Omean, D):
        p10, p90 = _pct_10_90(arr)
        feats.extend(
            [
                float(arr.mean()),
                float(arr.std()),
                float(np.median(arr)),
                p10,
                p90,
                float(arr.max()),
                float(arr.min()),
            ]
        )

    dt = D.mean(axis=1)  # (273,)
    df = D.mean(axis=0)  # (256,)
    feats.extend(
        [
            float(dt.mean()),
            float(dt.std()),
            float(dt.max()),
            float(dt.min()),
            float(df.mean()),
            float(df.std()),
            float(df.max()),
            float(df.min()),
        ]
    )

    feats.extend(
        [
            float((D * D).mean()),
            float(np.abs(D).mean()),
        ]
    )

    absD = np.abs(D)
    flat = absD.reshape(-1)
    k1 = 256
    k2 = 1024
    top1 = np.partition(flat, flat.size - k1)[-k1:]
    top2 = np.partition(flat, flat.size - k2)[-k2:]
    feats.extend(
        [
            float(top1.mean()),
            float(top1.max()),
            float(top2.mean()),
        ]
    )

    d_t = np.diff(D, axis=0)
    d_f = np.diff(D, axis=1)
    feats.extend(
        [
            float(np.mean(d_t * d_t)),
            float(np.mean(d_f * d_f)),
            float(np.mean(np.abs(d_t))),
            float(np.mean(np.abs(d_f))),
        ]
    )

    tmax = D.max(axis=1)
    dtmax = np.diff(tmax)
    feats.extend(
        [
            float(tmax.mean()),
            float(tmax.std()),
            float(dtmax.max()) if dtmax.size else 0.0,
            float(dtmax.min()) if dtmax.size else 0.0,
        ]
    )

    At = Amean.mean(axis=1)  # (273,)
    Ot = Omean.mean(axis=1)  # (273,)
    Af = Amean.mean(axis=0)  # (256,)
    Of = Omean.mean(axis=0)  # (256,)

    feats.extend(
        [
            float(At.std()),
            float(Ot.std()),
            float(At.max() - At.min()),
            float(Ot.max() - Ot.min()),
            float(Af.std()),
            float(Of.std()),
            float(Af.max() - Af.min()),
            float(Of.max() - Of.min()),
        ]
    )

    def _conc_feats(arr2d: np.ndarray) -> list:
        absx = np.abs(arr2d)
        s = float(absx.sum()) + 1e-8

        m_t = absx.max(axis=1)  # (273,)
        m_f = absx.max(axis=0)  # (256,)

        flatx = absx.reshape(-1)
        k_small = 256
        top_small = np.partition(flatx, flatx.size - k_small)[-k_small:]
        top_small_sum = float(top_small.sum())

        return [
            float(m_t.mean()),
            float(m_t.std()),
            float(m_t.max()),
            float(m_f.mean()),
            float(m_f.std()),
            float(m_f.max()),
            float(top_small_sum / s),
            float(float(flatx.max()) / s),
        ]

    feats.extend(_conc_feats(Amean))
    feats.extend(_conc_feats(Omean))
    feats.extend(_conc_feats(D))

    absA = np.abs(Amean)
    absO = np.abs(Omean)
    sumA = float(absA.sum()) + 1e-8
    sumO = float(absO.sum()) + 1e-8

    maxA = float(absA.max())
    maxO = float(absO.max())
    feats.extend(
        [
            float(maxA - maxO),
            float((maxA / sumA) - (maxO / sumO)),
        ]
    )

    A_tmax = absA.max(axis=1)
    O_tmax = absO.max(axis=1)
    A_fmax = absA.max(axis=0)
    O_fmax = absO.max(axis=0)
    feats.extend(
        [
            float(A_tmax.mean() - O_tmax.mean()),
            float(A_tmax.max() - O_tmax.max()),
            float(A_fmax.mean() - O_fmax.mean()),
            float(A_fmax.max() - O_fmax.max()),
        ]
    )

    flatA = absA.reshape(-1)
    flatO = absO.reshape(-1)
    k_mass = 1024
    topA = np.partition(flatA, flatA.size - k_mass)[-k_mass:]
    topO = np.partition(flatO, flatO.size - k_mass)[-k_mass:]
    feats.extend(
        [
            float((float(topA.sum()) / sumA) - (float(topO.sum()) / sumO)),
        ]
    )

    A_tproj = Amean.max(axis=1)  # (273,)
    O_tproj = Omean.max(axis=1)  # (273,)
    A_fproj = Amean.max(axis=0)  # (256,)
    O_fproj = Omean.max(axis=0)  # (256,)
    Dt = dt
    Df = df
    for arr1d in (A_tproj - O_tproj, A_fproj - O_fproj, Dt, Df):
        p10, p90 = _pct_10_90_1d(arr1d)
        feats.extend(
            [
                float(arr1d.mean()),
                float(arr1d.std()),
                float(p90 - p10),
                float(arr1d.max()),
                float(arr1d.min()),
            ]
        )

    Astd = A.std(axis=0)
    Ostd = O.std(axis=0)
    eps = 1e-8
    relA = Astd / (np.abs(Amean) + eps)
    relO = Ostd / (np.abs(Omean) + eps)
    relD = relO - relA

    for arr2d in (Astd, Ostd, relD):
        q90 = float(np.quantile(arr2d, 0.9, method="linear"))
        feats.extend(
            [
                float(arr2d.mean()),
                float(arr2d.std()),
                q90,
                float(arr2d.max()),
            ]
        )

    relDt = relD.mean(axis=1)  # (273,)
    relDf = relD.mean(axis=0)  # (256,)
    feats.extend(
        [
            float(relDt.mean()),
            float(relDt.std()),
            float(relDt.max()),
            float(relDf.mean()),
            float(relDf.std()),
            float(relDf.max()),
        ]
    )

    return np.array(feats, dtype=np.float32)


@lru_cache(maxsize=8192)
def _load_npy_cached(path: str) -> np.ndarray:
    arr = np.load(path)
    try:
        arr.setflags(write=False)
    except Exception:
        pass
    return arr


def featurize_ids(ids, id_to_path):
    ids = list(ids)

    _probe_arr = _load_npy_cached(id_to_path[ids[0]])
    n_feats = int(extract_features_from_array(_probe_arr.view()).shape[0])

    X = np.zeros((len(ids), n_feats), dtype=np.float32)

    def _one(i):
        fid = ids[i]
        arr = _load_npy_cached(id_to_path[fid])
        return i, extract_features_from_array(arr.view())

    max_workers = min(16, (os.cpu_count() or 4) * 2)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, feats in ex.map(_one, range(len(ids))):
            X[i] = feats

    return X


some_id = train_df.loc[0, "id"]
tmpX = featurize_ids([some_id], train_map)
tmpX.shape, tmpX[0, :5]



## === cell 4
X = featurize_ids(train_ids, train_map)
y = train_df["target"].astype(int).values

assert len(train_ids) == len(y) == X.shape[0]
assert (
    train_df["id"].tolist() == train_ids
), "Final alignment check failed (id order mismatch)."

groups = train_df["id"].str.slice(0, 3).values
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, va_idx = next(gss.split(X, y, groups=groups))
X_tr, X_va = X[tr_idx], X[va_idx]
y_tr, y_va = y[tr_idx], y[va_idx]

model = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                solver="lbfgs",
                max_iter=800,
                n_jobs=None,
                C=3.0,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]
)

model.fit(X_tr, y_tr)

va_pred = model.predict_proba(X_va)[:, 1]
print("Holdout AUC:", roc_auc_score(y_va, va_pred))



## === cell 5
model.fit(X, y)

test_ids = sample_sub["id"].tolist()

X_test = featurize_ids(test_ids, test_map)
pred = model.predict_proba(X_test)[:, 1].astype(np.float64)

eps = 1e-12
pred = np.clip(pred, eps, 1 - eps)

sub = pd.DataFrame({"id": test_ids, "target": pred})
sub.to_csv("submission.csv", index=False)

sub.head(), sub.shape
