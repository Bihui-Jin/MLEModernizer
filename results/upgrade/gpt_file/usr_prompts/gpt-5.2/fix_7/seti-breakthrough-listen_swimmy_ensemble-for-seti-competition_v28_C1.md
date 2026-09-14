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

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(0)




## === cell 1
def try_load_submission(path: str) -> pd.DataFrame | None:
    if os.path.exists(path):
        df = pd.read_csv(path)
        if not {"id", "target"}.issubset(df.columns):
            raise ValueError(
                f"{path} exists but does not have required columns id,target. Found: {df.columns.tolist()}"
            )
        return df[["id", "target"]].copy()
    return None


sample_paths = [
    "/kaggle/input/seti-breakthrough-listen/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
    "../input/seti-breakthrough-listen/sample_submission.csv",
    "../input/sample_submission.csv",
]
sample_path = next((p for p in sample_paths if os.path.exists(p)), None)
if sample_path is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv in expected Kaggle input locations."
    )

sample = pd.read_csv(sample_path)
if not {"id", "target"}.issubset(sample.columns):
    raise ValueError(
        f"sample_submission.csv missing required columns. Found: {sample.columns.tolist()}"
    )
sample = sample[["id", "target"]].copy()



## === cell 2
paths = {
    "data1": "../input/rerun-seti-e-t-volo-d1-baseline-inference/submission.csv",
    "data2": "../input/seti-bl-spatial-info-tf-tpu/submission.csv",
    "data3": "../input/seti-bl-tf-starter-tpu/submission.csv",
    "data4": "../input/seti-learned-image-resizing/submission.csv",
    "data5": "../input/lb-0-980-efficientnet-b0-more-epoch/submission.csv",
    "data6": "../input/inference-5x-ensemble-vanilla-resnet34d-seti/submission.csv",
}

loaded = {}
for k, p in paths.items():
    loaded[k] = try_load_submission(p)

available = [k for k, v in loaded.items() if v is not None]
print(f"Found {len(available)} external submission(s): {available}")




## === cell 3
def align_to_sample(df: pd.DataFrame, sample_ids: pd.Series) -> pd.DataFrame:
    out = sample_ids.to_frame(name="id").merge(
        df, on="id", how="left", validate="one_to_one"
    )
    if out["target"].isna().any():
        out["target"] = out["target"].fillna(0.5)
    return out


aligned = {}
for k, df in loaded.items():
    if df is not None:
        aligned[k] = align_to_sample(df, sample["id"])



## === cell 4
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


def find_existing_dir(candidates):
    for p in candidates:
        if os.path.isdir(p):
            return p
    return None


train_dir = find_existing_dir(
    [
        "/kaggle/input/seti-breakthrough-listen/train",
        "/kaggle/input/train",
        "../input/seti-breakthrough-listen/train",
        "../input/train",
    ]
)
test_dir = find_existing_dir(
    [
        "/kaggle/input/seti-breakthrough-listen/test",
        "/kaggle/input/test",
        "../input/seti-breakthrough-listen/test",
        "../input/test",
    ]
)
labels_path = next(
    (
        p
        for p in [
            "/kaggle/input/seti-breakthrough-listen/train_labels.csv",
            "/kaggle/input/train_labels.csv",
            "../input/seti-breakthrough-listen/train_labels.csv",
            "../input/train_labels.csv",
        ]
        if os.path.exists(p)
    ),
    None,
)

if train_dir is None or test_dir is None or labels_path is None:
    raise FileNotFoundError(
        f"Could not locate required train/test directories and/or train_labels.csv. "
        f"train_dir={train_dir}, test_dir={test_dir}, labels_path={labels_path}"
    )

labels = pd.read_csv(labels_path)
if not {"id", "target"}.issubset(labels.columns):
    raise ValueError(
        f"train_labels.csv missing required columns. Found: {labels.columns.tolist()}"
    )
labels = labels[["id", "target"]].copy()


def id_to_path(root: str, _id: str) -> str:
    return os.path.join(root, _id[0], f"{_id}.npy")


def _quantiles_linear_from_sorted(
    x_sorted_2d: np.ndarray, qs: np.ndarray
) -> np.ndarray:
    """
    x_sorted_2d: (n, m) sorted ascending, float32
    qs: (k,) float64/float32 in [0,1]
    returns: (k, n) float32 to match previous helper shape convention
    """
    n, m = x_sorted_2d.shape
    h = (m - 1) * qs.astype(np.float64, copy=False)
    lo = np.floor(h).astype(np.int64)
    hi = np.ceil(h).astype(np.int64)
    w = (h - lo).astype(np.float32)

    out = np.empty((len(qs), n), dtype=np.float32)
    for qi in range(len(qs)):
        klo = int(lo[qi])
        khi = int(hi[qi])
        if klo == khi:
            out[qi] = x_sorted_2d[:, klo]
        else:
            v_lo = x_sorted_2d[:, klo]
            v_hi = x_sorted_2d[:, khi]
            out[qi] = v_lo + (v_hi - v_lo) * w[qi]
    return out


def _quantiles_linear_small_axis(x_2d: np.ndarray, qs: np.ndarray) -> np.ndarray:
    """
    x_2d: (n, m) float32
    qs: (k,) float64/float32
    returns: (k, n) float32
    """
    x_sorted = np.sort(x_2d, axis=1)
    return _quantiles_linear_from_sorted(x_sorted, qs)


def build_feature_matrix(
    ids: np.ndarray,
    root: str,
    cache_path: str,
    n_features: int = 20,
    batch_size: int = 256,
) -> np.ndarray:
    if os.path.exists(cache_path):
        X = np.load(cache_path)
        if X.shape == (len(ids), n_features) and X.dtype == np.float32:
            return X

    X = np.empty((len(ids), n_features), dtype=np.float32)

    q3 = np.asarray((0.05, 0.50, 0.95), dtype=np.float64)
    q2 = np.asarray((0.05, 0.95), dtype=np.float64)

    xb = np.empty((batch_size, 6, 273, 256), dtype=np.float32)

    for start in range(0, len(ids), batch_size):
        end = min(len(ids), start + batch_size)
        bsz = end - start

        for i in range(bsz):
            _id = ids[start + i]
            p = id_to_path(root, _id)
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing npy file: {p}")
            arr = np.load(p, mmap_mode="r")
            if arr.dtype == np.float32:
                xb[i] = arr
            else:
                xb[i] = arr.astype(np.float32, copy=False)

        xbb = xb[:bsz]  # view

        a = xbb[:, [0, 2, 4], :, :]
        b = xbb[:, [1, 3, 5], :, :]
        da = a.mean(axis=1) - b.mean(axis=1)  # (bsz, 273, 256)

        xb_flat = xbb.reshape(bsz, -1)
        da_flat = da.reshape(bsz, -1)

        x_mean = xb_flat.mean(axis=1)
        x_std = xb_flat.std(axis=1)
        x_abs_mean = np.abs(xb_flat).mean(axis=1)

        xb_sorted = np.sort(xb_flat, axis=1)
        x_q = _quantiles_linear_from_sorted(xb_sorted, q3).T  # (bsz, 3)

        da_mean = da_flat.mean(axis=1)
        da_std = da_flat.std(axis=1)
        da_abs_mean = np.abs(da_flat).mean(axis=1)

        da_sorted = np.sort(da_flat, axis=1)
        da_q = _quantiles_linear_from_sorted(da_sorted, q3).T  # (bsz, 3)

        da_row = da.mean(axis=2)  # (bsz, 273)
        da_col = da.mean(axis=1)  # (bsz, 256)

        da_row_mean = da_row.mean(axis=1)
        da_row_std = da_row.std(axis=1)
        da_row_q = _quantiles_linear_small_axis(da_row, q2).T.astype(
            np.float32, copy=False
        )

        da_col_mean = da_col.mean(axis=1)
        da_col_std = da_col.std(axis=1)
        da_col_q = _quantiles_linear_small_axis(da_col, q2).T.astype(
            np.float32, copy=False
        )

        feats = np.column_stack(
            [
                x_mean,
                x_std,
                x_abs_mean,
                x_q[:, 0],
                x_q[:, 1],
                x_q[:, 2],
                da_mean,
                da_std,
                da_abs_mean,
                da_q[:, 0],
                da_q[:, 1],
                da_q[:, 2],
                da_row_mean,
                da_row_std,
                da_row_q[:, 0],
                da_row_q[:, 1],
                da_col_mean,
                da_col_std,
                da_col_q[:, 0],
                da_col_q[:, 1],
            ]
        ).astype(np.float32, copy=False)

        X[start:end] = feats

    np.save(cache_path, X)
    return X


train_ids = labels["id"].values
y = labels["target"].values.astype(np.int32)

X = build_feature_matrix(train_ids, train_dir, cache_path="X_train_feats.npy")

clf = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=200, solver="lbfgs", n_jobs=None)),
    ]
)

clf.fit(X, y)

test_ids = sample["id"].values
X_test = build_feature_matrix(test_ids, test_dir, cache_path="X_test_feats.npy")
baseline_pred = clf.predict_proba(X_test)[:, 1].astype(np.float64)

baseline = sample.copy()
baseline["target"] = np.clip(baseline_pred, 0.0, 1.0)

print("Baseline predictions ready:", baseline.shape, baseline["target"].describe())



## === cell 5
if {"data1", "data4", "data5"}.issubset(aligned.keys()):
    data1 = aligned["data1"]
    data4 = aligned.get(
        "data4", align_to_sample(sample, sample["id"])
    )  # should exist due to check
    data5 = aligned.get(
        "data5", align_to_sample(sample, sample["id"])
    )  # should exist due to check
    data6 = aligned.get("data6", data1).copy()

    data2 = aligned.get("data2", align_to_sample(sample, sample["id"]))
    data3 = aligned.get("data3", align_to_sample(sample, sample["id"]))

    data6["target"] = (
        0.77 * data5["target"].astype(float)
        + 0.14 * data4["target"].astype(float)
        + 0.09 * data6["target"].astype(float)
        + 0.0 * data2["target"].astype(float)
        + 0.0 * data3["target"].astype(float)
    )
    data6["target"] = data6["target"].clip(0.0, 1.0)
else:
    data6 = baseline.copy()

data6.head()



## === cell 6
data6 = data6[["id", "target"]]
data6.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", data6.shape)
print(data6.head())
