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

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42

BASE_INPUT = "/kaggle/input"
TRAIN_DIR = os.path.join(BASE_INPUT, "train")
TEST_DIR = os.path.join(BASE_INPUT, "test")
TRAIN_LABELS_PATH = os.path.join(BASE_INPUT, "train_labels.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")

assert os.path.exists(TRAIN_LABELS_PATH), f"Missing: {TRAIN_LABELS_PATH}"
assert os.path.exists(SAMPLE_SUB_PATH), f"Missing: {SAMPLE_SUB_PATH}"
assert os.path.isdir(TRAIN_DIR), f"Missing dir: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing dir: {TEST_DIR}"

train_labels = pd.read_csv(TRAIN_LABELS_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

train_labels.head(), sample_sub.head()




## === cell 1
def _id_from_path(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]


def list_npy_ids_and_paths(root_dir: str) -> pd.DataFrame:
    ids = []
    paths = []
    with os.scandir(root_dir) as it:
        for entry in it:
            if not entry.is_dir():
                continue
            with os.scandir(entry.path) as it2:
                for f in it2:
                    if f.is_file() and f.name.endswith(".npy"):
                        paths.append(f.path)
                        ids.append(os.path.splitext(f.name)[0])
    return pd.DataFrame({"id": ids, "path": paths})[["id", "path"]]


train_files = list_npy_ids_and_paths(TRAIN_DIR)
test_files = list_npy_ids_and_paths(TEST_DIR)

train_df = train_labels.merge(train_files, on="id", how="inner")
assert len(train_df) > 0, "No training files matched to labels."

test_df = sample_sub[["id"]].merge(test_files, on="id", how="left")
missing_test = test_df["path"].isna().sum()
assert (
    missing_test == 0
), f"Missing {missing_test} test .npy files referenced by sample_submission."

len(train_df), len(test_df), train_df.head()



## === cell 2
from joblib import Parallel, delayed

N_FEATURES = 93 + 30 + 20  # 143

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _cube_stats_10(cube2d: np.ndarray) -> np.ndarray:
    """
    Compute 10 stats over a 2D cube (H,W).
    Returns: (10,) float32 [mean,std,max,q10,q50,q90,grad_t,grad_f,drift,peak_over_q90]
    """
    x = cube2d.astype(np.float32, copy=False)

    mean = x.mean(dtype=np.float32)
    std = x.std(dtype=np.float32)
    mx = x.max()

    q10, q50, q90 = np.quantile(x, (0.10, 0.50, 0.90), method="linear").astype(
        np.float32
    )

    grad_t = np.abs(x[1:, :] - x[:-1, :]).mean(dtype=np.float32)
    grad_f = np.abs(x[:, 1:] - x[:, :-1]).mean(dtype=np.float32)

    idx = np.argmax(x, axis=1).astype(np.int32)  # (H,)
    drift = np.abs(np.diff(idx)).mean(dtype=np.float32)

    peak_over_q90 = np.float32(mx) - np.float32(q90)

    return np.array(
        [
            mean,
            std,
            np.float32(mx),
            q10,
            q50,
            q90,
            grad_t,
            grad_f,
            drift,
            peak_over_q90,
        ],
        dtype=np.float32,
    )


def extract_features_from_snippet(arr: np.ndarray) -> np.ndarray:
    x = arr.astype(np.float32, copy=False)
    if x.ndim != 3 or x.shape[0] != 6:
        raise ValueError(f"Unexpected snippet shape: {x.shape}, expected (6, H, W)")

    mean = x.mean(axis=(1, 2), dtype=np.float32)  # (6,)
    std = x.std(axis=(1, 2), dtype=np.float32)  # (6,)
    mx = x.max(axis=(1, 2)).astype(np.float32)  # (6,)

    q = np.quantile(x, (0.10, 0.50, 0.90), axis=(1, 2), method="linear").astype(
        np.float32
    )  # (3,6)
    q10, q50, q90 = q[0], q[1], q[2]

    grad_t = np.abs(x[:, 1:, :] - x[:, :-1, :]).mean(axis=(1, 2), dtype=np.float32)
    grad_f = np.abs(x[:, :, 1:] - x[:, :, :-1]).mean(axis=(1, 2), dtype=np.float32)

    idx = np.argmax(x, axis=2).astype(np.int32)  # (6, H)
    drift = np.abs(np.diff(idx, axis=1)).mean(axis=1).astype(np.float32)  # (6,)

    peak_over_q90 = (mx - q90).astype(np.float32)

    per_panel = np.stack(
        [mean, std, mx, q10, q50, q90, grad_t, grad_f, drift, peak_over_q90],
        axis=1,
    ).astype(
        np.float32, copy=False
    )  # (6,10)

    panel_mean = mean

    A_idx = np.array([0, 2, 4], dtype=np.int32)
    O_idx = np.array([1, 3, 5], dtype=np.int32)

    A_agg = per_panel[A_idx].mean(axis=0, dtype=np.float32)  # (10,)
    O_agg = per_panel[O_idx].mean(axis=0, dtype=np.float32)  # (10,)
    contrast = A_agg - O_agg

    d01 = np.float32(panel_mean[0] - panel_mean[1])
    d23 = np.float32(panel_mean[2] - panel_mean[3])
    d45 = np.float32(panel_mean[4] - panel_mean[5])

    diff01 = _cube_stats_10(x[0] - x[1])
    diff23 = _cube_stats_10(x[2] - x[3])
    diff45 = _cube_stats_10(x[4] - x[5])

    A_mean_cube = x[A_idx].mean(axis=0, dtype=np.float32)
    O_mean_cube = x[O_idx].mean(axis=0, dtype=np.float32)
    mean_AO_stats = _cube_stats_10(A_mean_cube - O_mean_cube)

    A_std_cube = x[A_idx].std(axis=0, dtype=np.float32)
    O_std_cube = x[O_idx].std(axis=0, dtype=np.float32)
    std_AO_stats = _cube_stats_10(A_std_cube - O_std_cube)

    feats = np.concatenate(
        [
            per_panel.reshape(-1),  # 60
            A_agg,
            O_agg,
            contrast,  # 30
            np.array([d01, d23, d45], dtype=np.float32),  # 3
            diff01,
            diff23,
            diff45,  # 30
            mean_AO_stats,
            std_AO_stats,  # 20
        ],
        axis=0,
    ).astype(np.float32, copy=False)

    if feats.shape[0] != N_FEATURES:
        raise RuntimeError(f"Feature length mismatch: {feats.shape[0]} != {N_FEATURES}")
    return feats


def _feats_from_path(p: str) -> np.ndarray:
    arr = np.load(p, mmap_mode="r")
    return extract_features_from_snippet(arr)


def build_feature_matrix(
    df: pd.DataFrame,
    path_col: str = "path",
    verbose_every: int = 5000,
    n_jobs: int = -1,
) -> np.ndarray:
    paths = df[path_col].to_numpy()
    n = len(paths)

    if verbose_every:
        print(f"Building features for {n} files...")

    feats_list = Parallel(n_jobs=n_jobs, prefer="threads", batch_size=64)(
        delayed(_feats_from_path)(p) for p in paths
    )
    X = np.stack(feats_list, axis=0).astype(np.float32, copy=False)
    return X


_X_smoke = build_feature_matrix(train_df.head(2), verbose_every=0, n_jobs=1)
_X_smoke.shape, _X_smoke[0][:10]



## === cell 3
X_train = build_feature_matrix(train_df, verbose_every=0, n_jobs=-1)
y_train = train_df["target"].astype(int).values

X_train.shape, y_train.mean()



## === cell 4
from sklearn.metrics import roc_auc_score

C_GRID = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

oof_pred = np.zeros(len(train_df), dtype=np.float32)
test_models = []
selected_Cs = []


def _eval_C_for_fold(C, X_tr, y_tr, inner_skf, seed_base):
    oof_inner = np.zeros(len(y_tr), dtype=np.float32)
    for inner_fold, (itr, iva) in enumerate(inner_skf.split(X_tr, y_tr), start=1):
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        solver="saga",
                        penalty="l2",
                        C=C,
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=seed_base + inner_fold,
                        n_jobs=1,  # avoid oversubscription inside joblib parallelism
                    ),
                ),
            ]
        )
        model.fit(X_tr[itr], y_tr[itr])
        oof_inner[iva] = model.predict_proba(X_tr[iva])[:, 1].astype(np.float32)
    auc = roc_auc_score(y_tr, oof_inner)
    return (C, float(auc))


for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), start=1):
    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_va, y_va = X_train[va_idx], y_train[va_idx]

    inner_skf = StratifiedKFold(
        n_splits=3, shuffle=True, random_state=RANDOM_STATE + 1000 * fold
    )

    seed_base = RANDOM_STATE + 1000 * fold

    results = Parallel(n_jobs=-1, prefer="processes", batch_size=1)(
        delayed(_eval_C_for_fold)(C, X_tr, y_tr, inner_skf, seed_base) for C in C_GRID
    )
    auc_by_C = {C: auc for C, auc in results}
    best_C = max(C_GRID, key=lambda c: (auc_by_C[c], -C_GRID.index(c)))

    selected_Cs.append(best_C)

    final_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="saga",
                    penalty="l2",
                    C=best_C,
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE + fold,
                    n_jobs=1,
                ),
            ),
        ]
    )
    final_model.fit(X_tr, y_tr)
    oof_pred[va_idx] = final_model.predict_proba(X_va)[:, 1].astype(np.float32)
    test_models.append(final_model)

print(
    "Per-fold selected Cs:",
    selected_Cs,
    "| OOF AUC:",
    float(roc_auc_score(y_train, oof_pred)),
)
print(
    "OOF predictions summary:",
    float(oof_pred.min()),
    float(oof_pred.max()),
    float(oof_pred.mean()),
)



## === cell 5
X_test = build_feature_matrix(test_df, verbose_every=0, n_jobs=-1)

test_pred = np.zeros(len(test_df), dtype=np.float32)
for model in test_models:
    test_pred += model.predict_proba(X_test)[:, 1].astype(np.float32)
test_pred /= len(test_models)

submission = sample_sub[["id"]].copy()
submission["target"] = test_pred
submission["target"] = submission["target"].clip(0.0, 1.0)

submission.head(), submission.shape



## === cell 6
out_path = "submission.csv"
submission.to_csv(out_path, index=False)

assert out_path.endswith(".csv")
assert list(submission.columns) == ["id", "target"]
assert len(submission) == len(sample_sub)

print(f"Wrote {out_path} with shape {submission.shape}")
print(submission.describe(include="all"))
