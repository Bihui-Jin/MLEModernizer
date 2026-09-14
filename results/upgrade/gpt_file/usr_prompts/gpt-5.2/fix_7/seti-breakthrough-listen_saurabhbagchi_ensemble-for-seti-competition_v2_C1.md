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

0.7265060115264068

# 6. Current score

0.49373

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.49684) has done: 'Your current notebook tries to ensemble three external Kaggle notebook submissions that aren’t available in this environment, causing a `FileNotFoundError` and preventing any `submission.csv` from being written. I replace that with a self-contained pipeline that reads the provided `train_labels.csv` and the `.npy` snippets under `train/` and `test/`, builds a lightweight feature set from each snippet (keeping the overall “simple baseline” spirit), trains a scikit-learn model, and writes a valid `submission.csv` with `id,target`. This run end-to-end using only the installed packages and the given data paths, and should yield a reasonable AUC toward your target. I also make sure ID ordering matches `sample_submission.csv` to avoid alignment bugs.'
- What this solution (achieved 0.49208) has done: 'Your timeout is almost entirely from Python-level per-file feature extraction: you load ~54k train + 6k test `.npy` files and, for each of 6 panels, call `np.quantile`, `np.diff`, and `argmax` separately—this creates a lot of repeated overhead. I keep the exact same features and model/5-fold training semantics, but make extraction equivalent and much faster by (1) vectorizing panel feature computation across all 6 panels at once, (2) replacing `np.quantile` with the equivalent `np.percentile` along fixed axes, (3) minimizing temporary arrays and type conversions, and (4) using parallel, deterministic feature-matrix building with `joblib` threads (NumPy releases the GIL for these ops, and `.npy` loading benefits from concurrency). The resulting feature values are the same up to negligible float rounding differences, and the training loop/model remains unchanged.'
- What this solution (achieved 0.49374) has done: 'Your current score (0.49208) is far below the target (0.7265), so we should improve generalization without changing the core “feature extraction + logistic regression + 5-fold CV + mean ensemble” logic. The smallest high-impact change here is to fix a likely regularization/optimization mismatch by using `class_weight="balanced"` (the dataset is typically imbalanced) and a slightly stronger, more stable solver for this feature set (`liblinear`), while keeping the same model family and prediction semantics. I also make the CV ensemble actually use fold-specific randomness by varying the `random_state` per fold (still deterministic overall), which can modestly improve the averaged test prediction. Everything else (features, folds, pipeline, submission alignment) stays the same.'
- What this solution (achieved 0.49216) has done: 'Your current gap to target is large (0.49374 → 0.7265), so we should improve discrimination while keeping the same “handcrafted features + StandardScaler + LogisticRegression + 5-fold mean ensemble” core. The smallest, high-impact tweak within the same model family is to switch LogisticRegression to the `saga` solver with L1 regularization (`penalty="l1"`), which can sparsify/denoise this 93-feature set and often yields a meaningful AUC lift without changing evaluation semantics. I also set a slightly higher `max_iter` for reliable convergence and keep determinism via fixed per-fold `random_state`. Everything else (features, folds, submission alignment/format) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 0.49373) has done: 'Your current score (0.49216) is far below the target (0.7265), so we should increase discrimination while keeping the same “handcrafted features + StandardScaler + LogisticRegression + 5-fold mean ensemble” core. The smallest high-impact fix is to undo the L1+saga change (which is likely over-sparsifying this small 93-feature set) and go back to a more stable, better-calibrated L2 logistic regression (`lbfgs`) while preserving the exact same training loop and prediction averaging. To keep this change minimal and controlled, I only adjust the logistic regression hyperparameters (penalty/solver/C/max_iter) and keep class balancing, folds, features, and submission alignment identical. This should move AUC upward toward your target without changing evaluation semantics.'

# 9. Code solution

## === cell 0
import os
import glob
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
    paths = glob.glob(os.path.join(root_dir, "*", "*.npy"))
    df = pd.DataFrame({"path": paths})
    df["id"] = df["path"].map(_id_from_path)
    return df[["id", "path"]]


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

N_FEATURES = 93  # must match concatenation below


def extract_features_from_snippet(arr: np.ndarray) -> np.ndarray:
    """
    arr: (6, 273, 256) float16/float32
    Same features as original: per-panel [mean,std,max,q10,q50,q90,grad_t,grad_f,drift,peak_over_q90],
    then A/O aggregates, contrast, and mean-differences (0-1,2-3,4-5).
    """
    x = arr.astype(np.float32, copy=False)
    if x.ndim != 3 or x.shape[0] != 6:
        raise ValueError(f"Unexpected snippet shape: {x.shape}, expected (6, H, W)")

    mean = x.mean(axis=(1, 2), dtype=np.float32)  # (6,)
    std = x.std(axis=(1, 2), dtype=np.float32)  # (6,)
    mx = x.max(axis=(1, 2))  # (6,)

    q = np.percentile(x, [10, 50, 90], axis=(1, 2)).astype(np.float32)  # (3,6)
    q10, q50, q90 = q[0], q[1], q[2]

    d_t = np.diff(x, axis=1)  # (6, H-1, W)
    d_f = np.diff(x, axis=2)  # (6, H, W-1)
    grad_t = np.abs(d_t).mean(axis=(1, 2), dtype=np.float32)
    grad_f = np.abs(d_f).mean(axis=(1, 2), dtype=np.float32)

    idx = np.argmax(x, axis=2).astype(np.int32)  # (6, H)
    drift = np.abs(np.diff(idx, axis=1)).mean(axis=1).astype(np.float32)  # (6,)

    peak_over_q90 = (mx - q90).astype(np.float32)

    per_panel = np.stack(
        [
            mean,
            std,
            mx.astype(np.float32),
            q10,
            q50,
            q90,
            grad_t,
            grad_f,
            drift,
            peak_over_q90,
        ],
        axis=1,
    ).astype(
        np.float32, copy=False
    )  # (6,10)

    panel_mean = per_panel[:, 0]

    A_idx = np.array([0, 2, 4])
    O_idx = np.array([1, 3, 5])
    A_agg = per_panel[A_idx].mean(axis=0, dtype=np.float32)  # (10,)
    O_agg = per_panel[O_idx].mean(axis=0, dtype=np.float32)  # (10,)
    contrast = A_agg - O_agg

    d01 = panel_mean[0] - panel_mean[1]
    d23 = panel_mean[2] - panel_mean[3]
    d45 = panel_mean[4] - panel_mean[5]

    feats = np.concatenate(
        [
            per_panel.reshape(-1),  # 60
            A_agg,  # 10
            O_agg,  # 10
            contrast,  # 10
            np.array([d01, d23, d45], dtype=np.float32),  # 3
        ]
    ).astype(np.float32, copy=False)

    if feats.shape[0] != N_FEATURES:
        raise RuntimeError(f"Feature length mismatch: {feats.shape[0]} != {N_FEATURES}")

    return feats


def _feats_from_path(p: str) -> np.ndarray:
    arr = np.load(p)
    return extract_features_from_snippet(arr)


def build_feature_matrix(
    df: pd.DataFrame,
    path_col: str = "path",
    verbose_every: int = 5000,
    n_jobs: int = -1,
) -> np.ndarray:
    paths = df[path_col].values
    n = len(paths)

    if verbose_every:
        print(f"Building features for {n} files...")

    feats_list = Parallel(n_jobs=n_jobs, prefer="threads", batch_size=64)(
        delayed(_feats_from_path)(p) for p in paths
    )
    X = np.vstack(feats_list).astype(np.float32, copy=False)
    return X


_X_smoke = build_feature_matrix(train_df.head(2), verbose_every=0, n_jobs=1)
_X_smoke.shape, _X_smoke[0][:10]



## === cell 3
X_train = build_feature_matrix(train_df, verbose_every=0, n_jobs=-1)
y_train = train_df["target"].astype(int).values

X_train.shape, y_train.mean()



## === cell 4
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

oof_pred = np.zeros(len(train_df), dtype=np.float32)
test_pred_folds = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), start=1):
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="lbfgs",
                    penalty="l2",
                    C=1.0,
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE + fold,  # deterministic but fold-specific
                ),
            ),
        ]
    )
    model.fit(X_train[tr_idx], y_train[tr_idx])
    oof_pred[va_idx] = model.predict_proba(X_train[va_idx])[:, 1].astype(np.float32)
    test_pred_folds.append(model)

print(
    "OOF predictions summary:",
    float(oof_pred.min()),
    float(oof_pred.max()),
    float(oof_pred.mean()),
)



## === cell 5
X_test = build_feature_matrix(test_df, verbose_every=0, n_jobs=-1)

test_pred = np.zeros(len(test_df), dtype=np.float32)
for model in test_pred_folds:
    test_pred += model.predict_proba(X_test)[:, 1].astype(np.float32)
test_pred /= len(test_pred_folds)

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
