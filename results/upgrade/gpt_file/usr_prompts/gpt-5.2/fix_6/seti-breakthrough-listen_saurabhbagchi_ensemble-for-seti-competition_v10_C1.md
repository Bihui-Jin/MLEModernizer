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



## === cell 1
BASE_PATH = "/kaggle/input"
TRAIN_LABELS_PATH = os.path.join(BASE_PATH, "train_labels.csv")
TRAIN_DIR = os.path.join(BASE_PATH, "train")
TEST_DIR = os.path.join(BASE_PATH, "test")
SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")

assert os.path.exists(TRAIN_LABELS_PATH), f"Missing: {TRAIN_LABELS_PATH}"
assert os.path.exists(TRAIN_DIR), f"Missing: {TRAIN_DIR}"
assert os.path.exists(TEST_DIR), f"Missing: {TEST_DIR}"
assert os.path.exists(SAMPLE_SUB_PATH), f"Missing: {SAMPLE_SUB_PATH}"

train_labels = pd.read_csv(TRAIN_LABELS_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

train_labels.head(), sample_sub.head()




## === cell 2
def id_to_path(root_dir: str, _id: str) -> str:
    """
    Change (score-improvement, same core logic): fix sharding to match the dataset layout.
    SETI data is commonly stored in folders 0..15 (hex shard) rather than '0'..'f'.
    We try the 0..15 path first; if it doesn't exist, fall back to '0'..'f' for robustness.
    """
    shard_int = int(_id[0], 16)
    p1 = os.path.join(root_dir, str(shard_int), f"{_id}.npy")
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(root_dir, _id[0], f"{_id}.npy")
    return p2


def extract_features_from_npy(path: str) -> np.ndarray:
    """
    Minimal, fast feature extraction from (6,273,256) float16 snippet.

    Change (score-improvement, same core logic): add a few additional A-vs-BCD contrast features
    that better capture "present in A (0,2,4) but absent in (1,3,5)" without changing the
    overall approach (still just NumPy reductions + logistic regression).
    """
    x = np.load(path, mmap_mode="r")  # (6, 273, 256)
    x = x.astype(np.float32, copy=False)

    mean = x.mean()
    std = x.std()
    mx = x.max()
    mn = x.min()

    a = x[[0, 2, 4]]
    bcd = x[[1, 3, 5]]
    a_mean = a.mean()
    b_mean = bcd.mean()
    a_std = a.std()
    b_std = bcd.std()

    xc = x - mean
    l2 = np.sqrt((xc * xc).mean())

    tprof = x.mean(axis=2)  # (6,273)
    fprof = x.mean(axis=1)  # (6,256)

    t_mean = tprof.mean()
    t_std = tprof.std()
    t_max = tprof.max()

    f_mean = fprof.mean()
    f_std = fprof.std()
    f_max = fprof.max()

    tA = tprof[[0, 2, 4]].mean(axis=0)  # (273,)
    tB = tprof[[1, 3, 5]].mean(axis=0)  # (273,)
    fA = fprof[[0, 2, 4]].mean(axis=0)  # (256,)
    fB = fprof[[1, 3, 5]].mean(axis=0)  # (256,)

    t_diff = tA - tB
    f_diff = fA - fB

    eps = np.float32(1e-6)
    t_diff_l2 = np.sqrt((t_diff * t_diff).mean())
    f_diff_l2 = np.sqrt((f_diff * f_diff).mean())
    t_diff_max = t_diff.max()
    t_diff_min = t_diff.min()
    f_diff_max = f_diff.max()
    f_diff_min = f_diff.min()

    a_over_b_mean = a_mean / (b_mean + eps)
    a_over_b_std = a_std / (b_std + eps)

    abs_contrast_mean = (a_mean - b_mean) / (std + eps)
    tdiff_pos_mass = np.mean(np.maximum(t_diff, 0.0))  # positive-only drift-like mass
    fdiff_pos_mass = np.mean(np.maximum(f_diff, 0.0))
    tA_peak = tA.max() - tA.mean()
    tB_peak = tB.max() - tB.mean()
    fA_peak = fA.max() - fA.mean()
    fB_peak = fB.max() - fB.mean()
    t_peak_diff = tA_peak - tB_peak
    f_peak_diff = fA_peak - fB_peak

    return np.array(
        [
            mean,
            std,
            mx,
            mn,
            a_mean,
            b_mean,
            a_std,
            b_std,
            a_mean - b_mean,
            l2,
            t_mean,
            t_std,
            t_max,
            f_mean,
            f_std,
            f_max,
            t_diff_l2,
            f_diff_l2,
            t_diff_max,
            t_diff_min,
            f_diff_max,
            f_diff_min,
            a_over_b_mean,
            a_over_b_std,
            abs_contrast_mean,
            tdiff_pos_mass,
            fdiff_pos_mass,
            t_peak_diff,
            f_peak_diff,
        ],
        dtype=np.float32,
    )




## === cell 3
from concurrent.futures import ProcessPoolExecutor


def _featurize_one(path: str) -> np.ndarray:
    return extract_features_from_npy(path)


train_ids = train_labels["id"].values
train_y = train_labels["target"].values

train_paths = [id_to_path(TRAIN_DIR, _id) for _id in train_ids]
exists_mask = np.fromiter(
    (os.path.exists(p) for p in train_paths), count=len(train_paths), dtype=bool
)

missing = int((~exists_mask).sum())
if missing > 0:
    print(
        f"Warning: missing {missing} training files. (Path/shard mismatch would hurt AUC.)"
    )

train_paths_ok = [p for p, ok in zip(train_paths, exists_mask) if ok]
y_ok = train_y[exists_mask]

max_workers = min(16, max(1, (os.cpu_count() or 4) // 2))
with ProcessPoolExecutor(max_workers=max_workers) as ex:
    X_list = list(ex.map(_featurize_one, train_paths_ok, chunksize=64))

X = np.vstack(X_list).astype(np.float32, copy=False)
y = np.array(y_ok, dtype=np.int64)

X.shape, y.shape, y.mean()



## === cell 4
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

SEED = 42

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

oof = np.zeros(len(y), dtype=np.float32)


def make_model():
    return Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            (
                "clf",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=500,
                    n_jobs=None,
                    random_state=SEED,
                    C=2.0,
                    class_weight="balanced",
                ),
            ),
        ]
    )


for tr_idx, va_idx in skf.split(X, y):
    model_fold = make_model()
    model_fold.fit(X[tr_idx], y[tr_idx])
    oof[va_idx] = model_fold.predict_proba(X[va_idx])[:, 1].astype(np.float32)

model = make_model()
model.fit(X, y)



## === cell 5
test_ids = sample_sub["id"].values
test_paths = [id_to_path(TEST_DIR, _id) for _id in test_ids]
exists_mask_t = np.fromiter(
    (os.path.exists(p) for p in test_paths), count=len(test_paths), dtype=bool
)

missing_test = int((~exists_mask_t).sum())
if missing_test > 0:
    print(
        f"Warning: missing {missing_test} test files. (Path/shard mismatch would hurt AUC.)"
    )

X_test = np.zeros((len(test_ids), X.shape[1]), dtype=np.float32)

idx_ok = np.flatnonzero(exists_mask_t)
paths_ok = [test_paths[i] for i in idx_ok]

with ProcessPoolExecutor(max_workers=max_workers) as ex:
    feats_ok = list(ex.map(_featurize_one, paths_ok, chunksize=64))

if len(idx_ok):
    X_test[idx_ok] = np.vstack(feats_ok).astype(np.float32, copy=False)

X_test.shape



## === cell 6
test_pred = model.predict_proba(X_test)[:, 1].astype(np.float32)

submission = pd.DataFrame({"id": test_ids, "target": test_pred})
submission = submission.merge(sample_sub[["id"]], on="id", how="right")
submission["target"] = (
    submission["target"].fillna(submission["target"].mean()).astype(float)
)

submission.to_csv("submission.csv", index=False)
submission.head(), submission.shape
