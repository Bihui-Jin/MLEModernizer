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

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

np.random.seed(0)



## === cell 1
TEST_DIR = "/kaggle/data/test"
TRAIN_DIR = "/kaggle/data/train"
TRAIN_LABELS_PATH = "/kaggle/data/train_labels.csv"
SAMPLE_SUB_PATH = "/kaggle/data/sample_submission.csv"

sample = pd.read_csv(SAMPLE_SUB_PATH)
sample["id"] = sample["id"].astype(str)

train_labels = pd.read_csv(TRAIN_LABELS_PATH)
train_labels["id"] = train_labels["id"].astype(str)

test_paths = sorted(glob.glob(os.path.join(TEST_DIR, "*", "*.npy")))
if len(test_paths) == 0:
    raise FileNotFoundError(
        f"No .npy files found under {TEST_DIR}. Check dataset path."
    )

train_paths = sorted(glob.glob(os.path.join(TRAIN_DIR, "*", "*.npy")))
if len(train_paths) == 0:
    raise FileNotFoundError(
        f"No .npy files found under {TRAIN_DIR}. Check dataset path."
    )

id_to_test_path = {os.path.splitext(os.path.basename(p))[0]: p for p in test_paths}
missing = [i for i in sample["id"].tolist() if i not in id_to_test_path]
if missing:
    raise FileNotFoundError(
        f"Missing {len(missing)} test .npy files for ids from sample_submission. Example: {missing[:5]}"
    )
ordered_test_paths = [id_to_test_path[i] for i in sample["id"].tolist()]

id_to_train_path = {os.path.splitext(os.path.basename(p))[0]: p for p in train_paths}
train_ids = train_labels["id"].tolist()
missing_tr = [i for i in train_ids if i not in id_to_train_path]
if missing_tr:
    raise FileNotFoundError(
        f"Missing {len(missing_tr)} train .npy files for ids from train_labels.csv. Example: {missing_tr[:5]}"
    )
ordered_train_paths = [id_to_train_path[i] for i in train_ids]


def _extract_features_from_snippet(arr_6_273_256: np.ndarray) -> dict:
    """
    Lightweight deterministic feature extraction from (6,273,256) snippet.
    Produces several different "views" to stand in for different model submissions.

    Changes here are score-driven but still minimal: add a few additional robust
    A-vs-B summary stats (quantiles/projections) that often improve AUC ranking.
    """
    x = arr_6_273_256.astype(np.float32)

    mean_all = float(x.mean())
    std_all = float(x.std() + 1e-6)

    A = x[[0, 2, 4]]
    B = x[[1, 3, 5]]

    mean_A = float(A.mean())
    mean_B = float(B.mean())
    std_A = float(A.std() + 1e-6)
    std_B = float(B.std() + 1e-6)

    eA = float((A * A).mean())
    eB = float((B * B).mean())
    energy_ratio = float(np.log((eA + 1e-6) / (eB + 1e-6)))

    d_t = float(np.mean(np.abs(np.diff(A, axis=1))))  # along time dimension (273)
    d_f = float(np.mean(np.abs(np.diff(A, axis=2))))  # along freq dimension (256)

    A_stack_mean = A.mean(axis=0)
    A_consistency = float(np.mean((A - A_stack_mean) ** 2))

    Aq90 = float(np.quantile(A, 0.90))
    Bq90 = float(np.quantile(B, 0.90))
    Aq99 = float(np.quantile(A, 0.99))
    Bq99 = float(np.quantile(B, 0.99))

    Amax = float(np.max(A))
    Bmax = float(np.max(B))

    A2 = A.mean(axis=0)  # (273,256)
    B2 = B.mean(axis=0)
    A_row = A2.mean(axis=1)  # (273,)
    A_col = A2.mean(axis=0)  # (256,)
    B_row = B2.mean(axis=1)
    B_col = B2.mean(axis=0)

    A_row_mad = float(np.mean(np.abs(A_row - A_row.mean())))
    A_col_mad = float(np.mean(np.abs(A_col - A_col.mean())))
    B_row_mad = float(np.mean(np.abs(B_row - B_row.mean())))
    B_col_mad = float(np.mean(np.abs(B_col - B_col.mean())))

    return {
        "mean_all": mean_all,
        "std_all": std_all,
        "mean_A": mean_A,
        "mean_B": mean_B,
        "std_A": std_A,
        "std_B": std_B,
        "energy_ratio": energy_ratio,
        "d_t": d_t,
        "d_f": d_f,
        "A_consistency": A_consistency,
        "Aq90": Aq90,
        "Bq90": Bq90,
        "Aq99": Aq99,
        "Bq99": Bq99,
        "Amax": Amax,
        "Bmax": Bmax,
        "A_row_mad": A_row_mad,
        "A_col_mad": A_col_mad,
        "B_row_mad": B_row_mad,
        "B_col_mad": B_col_mad,
    }


def _build_feature_df(ids_list, paths_list) -> pd.DataFrame:
    feats = []
    for i, p in zip(ids_list, paths_list):
        arr = np.load(p)
        if arr.shape != (6, 273, 256):
            raise ValueError(
                f"Unexpected shape for {i}: {arr.shape}, expected (6,273,256)"
            )
        f = _extract_features_from_snippet(arr)
        f["id"] = i
        feats.append(f)
    return pd.DataFrame(feats)


train_feat_df = _build_feature_df(train_ids, ordered_train_paths)
test_feat_df = _build_feature_df(sample["id"].tolist(), ordered_test_paths)


def _compute_z(df: pd.DataFrame):
    z1 = (df["mean_A"] - df["mean_B"]) / (df["std_all"])
    z2 = df["energy_ratio"]
    z3 = (df["d_t"] - df["d_f"]) / (df["std_A"])
    z4 = -np.log1p(df["A_consistency"])  # larger when more consistent (less variance)
    z5 = (df["mean_A"] - df["mean_all"]) / (df["std_all"])
    z6 = (df["std_A"] - df["std_B"]) / (df["std_all"])

    z7 = (df["Aq90"] - df["Bq90"]) / (df["std_all"])
    z8 = (df["Aq99"] - df["Bq99"]) / (df["std_all"])
    z9 = (df["Amax"] - df["Bmax"]) / (df["std_all"])
    z10 = np.log1p(df["A_row_mad"]) - np.log1p(df["B_row_mad"])
    z11 = np.log1p(df["A_col_mad"]) - np.log1p(df["B_col_mad"])
    return z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11


(tr_z1, tr_z2, tr_z3, tr_z4, tr_z5, tr_z6, tr_z7, tr_z8, tr_z9, tr_z10, tr_z11) = (
    _compute_z(train_feat_df)
)

(te_z1, te_z2, te_z3, te_z4, te_z5, te_z6, te_z7, te_z8, te_z9, te_z10, te_z11) = (
    _compute_z(test_feat_df)
)

X_train = np.vstack(
    [tr_z1, tr_z2, tr_z3, tr_z4, tr_z5, tr_z6, tr_z7, tr_z8, tr_z9, tr_z10, tr_z11]
).T.astype(np.float32)
y_train = train_labels["target"].values.astype(np.int32)
X_test = np.vstack(
    [te_z1, te_z2, te_z3, te_z4, te_z5, te_z6, te_z7, te_z8, te_z9, te_z10, te_z11]
).T.astype(np.float32)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train).astype(np.float32)
X_test_s = scaler.transform(X_test).astype(np.float32)

C_grid = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

best_C, best_auc = None, -1.0
for C in C_grid:
    aucs = []
    for tr_idx, va_idx in skf.split(X_train_s, y_train):
        tmp = LogisticRegression(
            solver="lbfgs",
            max_iter=800,
            C=C,
            class_weight=None,
            random_state=0,
        )
        tmp.fit(X_train_s[tr_idx], y_train[tr_idx])
        p_va = tmp.predict_proba(X_train_s[va_idx])[:, 1]
        aucs.append(roc_auc_score(y_train[va_idx], p_va))
    mean_auc = float(np.mean(aucs))
    if mean_auc > best_auc:
        best_auc = mean_auc
        best_C = C

oof = np.zeros(len(y_train), dtype=np.float32)
for tr_idx, va_idx in skf.split(X_train_s, y_train):
    tmp = LogisticRegression(
        solver="lbfgs",
        max_iter=800,
        C=best_C,
        class_weight=None,
        random_state=0,
    )
    tmp.fit(X_train_s[tr_idx], y_train[tr_idx])
    oof[va_idx] = tmp.predict_proba(X_train_s[va_idx])[:, 1].astype(np.float32)

clf = LogisticRegression(
    solver="lbfgs",
    max_iter=800,
    C=best_C,
    class_weight=None,
    random_state=0,
)
clf.fit(X_train_s, y_train)
p_test_raw = clf.predict_proba(X_test_s)[:, 1].astype(np.float32)

eps_clip = 1e-5
oof_clip = np.clip(oof, eps_clip, 1.0 - eps_clip)
test_clip = np.clip(p_test_raw, eps_clip, 1.0 - eps_clip)

oof_logit = np.log(oof_clip / (1.0 - oof_clip)).reshape(-1, 1).astype(np.float32)
test_logit = np.log(test_clip / (1.0 - test_clip)).reshape(-1, 1).astype(np.float32)

cal = LogisticRegression(
    solver="lbfgs",
    max_iter=800,
    C=1.0,
    class_weight=None,
    random_state=0,
)
cal.fit(oof_logit, y_train)
p_base = cal.predict_proba(test_logit)[:, 1].astype(np.float32)

print(f"Selected C={best_C} with 5-fold CV AUC={best_auc:.5f}")
print(f"OOF AUC at best_C (sanity): {roc_auc_score(y_train, oof):.5f}")

base_clip = np.clip(p_base, eps_clip, 1.0 - eps_clip)
base_logit = np.log(base_clip / (1.0 - base_clip)).astype(np.float32)

Z_te = np.vstack([te_z1, te_z2, te_z3, te_z4, te_z5, te_z6]).T.astype(np.float32)
Z_te_s = StandardScaler().fit_transform(Z_te).astype(np.float32)

eps = 0.012  # smaller than before; keeps ensemble diversity but reduces ranking-noise
l1 = base_logit + eps * np.tanh(Z_te_s[:, 0])
l2 = base_logit + eps * np.tanh(Z_te_s[:, 2])
l3 = base_logit + eps * np.tanh(Z_te_s[:, 1])
l4 = base_logit + eps * np.tanh(Z_te_s[:, 3])
l5 = base_logit + eps * np.tanh(Z_te_s[:, 4])
l6 = base_logit + eps * np.tanh(Z_te_s[:, 5])

sigmoid = lambda a: (1.0 / (1.0 + np.exp(-a))).astype(np.float32)

pred1 = np.clip(sigmoid(l1), 0.0, 1.0)
pred2 = np.clip(sigmoid(l2), 0.0, 1.0)
pred3 = np.clip(sigmoid(l3), 0.0, 1.0)
pred4 = np.clip(sigmoid(l4), 0.0, 1.0)
pred5 = np.clip(sigmoid(l5), 0.0, 1.0)
pred6 = np.clip(sigmoid(l6), 0.0, 1.0)

ids = sample["id"].tolist()
data1 = pd.DataFrame({"id": ids, "target": pred1.astype(np.float32)})
data2 = pd.DataFrame({"id": ids, "target": pred2.astype(np.float32)})
data3 = pd.DataFrame({"id": ids, "target": pred3.astype(np.float32)})
data4 = pd.DataFrame({"id": ids, "target": pred4.astype(np.float32)})
data5 = pd.DataFrame({"id": ids, "target": pred5.astype(np.float32)})
data6 = pd.DataFrame({"id": ids, "target": pred6.astype(np.float32)})



## === cell 2
data1.head()



## === cell 3
data2.head()



## === cell 4
m = data1[["id"]].copy()
m = m.merge(data1, on="id", how="left", suffixes=("", "_1"))
m = m.merge(data2, on="id", how="left", suffixes=("", "_2"))
m = m.merge(data3, on="id", how="left", suffixes=("", "_3"))
m = m.merge(data4, on="id", how="left", suffixes=("", "_4"))
m = m.merge(data5, on="id", how="left", suffixes=("", "_5"))
m = m.merge(data6, on="id", how="left", suffixes=("", "_6"))

w5, w4, w6, w2, w3 = 0.75, 0.115, 0.115, 0.1, 0.1
wsum = w5 + w4 + w6 + w2 + w3
w5, w4, w6, w2, w3 = (w5 / wsum, w4 / wsum, w6 / wsum, w2 / wsum, w3 / wsum)

data6 = m[["id"]].copy()
data6["target"] = (
    w5 * m["target_5"]
    + w4 * m["target_4"]
    + w6 * m["target_6"]
    + w2 * m["target_2"]
    + w3 * m["target_3"]
)

data6["target"] = data6["target"].clip(0.0, 1.0).astype(np.float32)



## === cell 5
data6.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", data6.shape)
print(data6.head())
