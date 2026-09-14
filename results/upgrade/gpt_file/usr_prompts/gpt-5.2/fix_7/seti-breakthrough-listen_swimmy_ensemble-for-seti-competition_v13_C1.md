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
BASE_DATA_DIRS = [
    "/kaggle/data",
    "/kaggle/input",
]

SAMPLE_SUB_PATH_CANDIDATES = [
    "/kaggle/data/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/seti-breakthrough-listen/sample_submission.csv",
    "/kaggle/input/seti-breakthrough-listen/sample_submission.csv",
]

TRAIN_LABELS_PATH_CANDIDATES = [
    "/kaggle/data/train_labels.csv",
    "/kaggle/input/train_labels.csv",
    "/kaggle/data/seti-breakthrough-listen/train_labels.csv",
    "/kaggle/input/seti-breakthrough-listen/train_labels.csv",
]

TRAIN_DIR_CANDIDATES = [
    "/kaggle/data/train",
    "/kaggle/input/train",
    "/kaggle/data/seti-breakthrough-listen/train",
    "/kaggle/input/seti-breakthrough-listen/train",
]

TEST_DIR_CANDIDATES = [
    "/kaggle/data/test",
    "/kaggle/input/test",
    "/kaggle/data/seti-breakthrough-listen/test",
    "/kaggle/input/seti-breakthrough-listen/test",
]


def find_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


sample_path = find_first_existing(SAMPLE_SUB_PATH_CANDIDATES)
if sample_path is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv in expected locations."
    )

sample = pd.read_csv(sample_path)
if not {"id", "target"}.issubset(sample.columns):
    raise ValueError(
        f"sample_submission.csv must have columns id,target; got {sample.columns.tolist()}"
    )

sample_ids = sample["id"].astype(str).tolist()
n_sample = len(sample_ids)

train_labels_path = find_first_existing(TRAIN_LABELS_PATH_CANDIDATES)
train_dir = find_first_existing(TRAIN_DIR_CANDIDATES)
test_dir = find_first_existing(TEST_DIR_CANDIDATES)

if train_labels_path is None:
    raise FileNotFoundError("Could not locate train_labels.csv in expected locations.")
if train_dir is None or not os.path.isdir(train_dir):
    raise FileNotFoundError("Could not locate train/ directory in expected locations.")
if test_dir is None or not os.path.isdir(test_dir):
    raise FileNotFoundError("Could not locate test/ directory in expected locations.")

labels = pd.read_csv(train_labels_path)
labels["id"] = labels["id"].astype(str)
if not {"id", "target"}.issubset(labels.columns):
    raise ValueError(
        f"train_labels.csv must have columns id,target; got {labels.columns.tolist()}"
    )

print("Using:")
print(" sample_submission:", sample_path)
print(" train_labels:", train_labels_path)
print(" train_dir:", train_dir)
print(" test_dir:", test_dir)



## === cell 2
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


def list_npy_files(root_dir):
    return sorted(glob.glob(os.path.join(root_dir, "**", "*.npy"), recursive=True))


def ids_from_paths(paths):
    return [os.path.splitext(os.path.basename(p))[0] for p in paths]


train_paths = list_npy_files(train_dir)
test_paths = list_npy_files(test_dir)

train_ids = ids_from_paths(train_paths)
test_ids = ids_from_paths(test_paths)

print(f"Found train npy files: {len(train_paths)}")
print(f"Found test  npy files: {len(test_paths)}")

train_path_by_id = {i: p for i, p in zip(train_ids, train_paths)}
test_path_by_id = {i: p for i, p in zip(test_ids, test_paths)}

labels = labels[labels["id"].isin(train_path_by_id)].reset_index(drop=True)
print(f"Train labels after path intersection: {len(labels)} (should be ~54000)")



## === cell 3
N_FEATS = 17


def extract_features_from_npy(path):
    """
    Minimal, stable feature extractor:
    - Uses aggregate statistics and A-vs-nonA differences, which are directly tied to the problem statement.
    - No architecture/training-loop change (sklearn linear model).
    """
    x = np.load(path)  # (6, 273, 256), float16
    x = x.astype(np.float32)

    A = x[[0, 2, 4]]
    B = x[[1, 3, 5]]

    feat = []
    feat.append(x.mean())
    feat.append(x.std() + 1e-6)
    feat.append(np.median(x))
    feat.append(np.quantile(x, 0.95))
    feat.append(np.quantile(x, 0.05))

    d = A.mean(axis=0) - B.mean(axis=0)
    feat.append(d.mean())
    feat.append(d.std() + 1e-6)
    feat.append(np.max(d))
    feat.append(np.min(d))
    feat.append(np.quantile(d, 0.95))
    feat.append(np.quantile(d, 0.05))

    k = 256
    A_flat = A.reshape(-1)
    B_flat = B.reshape(-1)
    feat.append(np.mean(np.partition(A_flat, -k)[-k:]))
    feat.append(np.mean(np.partition(B_flat, -k)[-k:]))

    A_mean = A.mean(axis=0)
    B_mean = B.mean(axis=0)
    feat.append(A_mean.mean(axis=1).std() + 1e-6)  # time-profile std
    feat.append(A_mean.mean(axis=0).std() + 1e-6)  # freq-profile std
    feat.append(B_mean.mean(axis=1).std() + 1e-6)
    feat.append(B_mean.mean(axis=0).std() + 1e-6)

    feat = np.array(feat, dtype=np.float32)

    if feat.shape[0] != N_FEATS:
        raise ValueError(
            f"Feature length mismatch: expected {N_FEATS}, got {feat.shape[0]} for path={path}"
        )

    return feat


X = np.zeros((len(labels), N_FEATS), dtype=np.float32)
y = labels["target"].to_numpy(dtype=np.int64)

for i, id_ in enumerate(labels["id"].tolist()):
    X[i] = extract_features_from_npy(train_path_by_id[id_])

print("Train feature matrix:", X.shape, "Positive rate:", float(y.mean()))

Xt = np.zeros((len(sample_ids), N_FEATS), dtype=np.float32)
missing_test = 0
for i, id_ in enumerate(sample_ids):
    p = test_path_by_id.get(id_)
    if p is None:
        missing_test += 1
        Xt[i] = 0.0
    else:
        Xt[i] = extract_features_from_npy(p)
print("Test feature matrix:", Xt.shape, "Missing test ids:", missing_test)



## === cell 4
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof = np.zeros(len(labels), dtype=np.float32)
test_pred = np.zeros(len(sample_ids), dtype=np.float32)

clf = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")),
    ]
)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
    Xtr, Xva = X[tr_idx], X[va_idx]
    ytr, yva = y[tr_idx], y[va_idx]

    clf.fit(Xtr, ytr)
    oof[va_idx] = clf.predict_proba(Xva)[:, 1].astype(np.float32)
    test_pred += clf.predict_proba(Xt)[:, 1].astype(np.float32) / skf.n_splits
    print(f"Finished fold {fold}")

test_pred = np.clip(test_pred, 0.0, 1.0)



## === cell 5
out_path = "submission.csv"
sub = pd.DataFrame({"id": sample_ids, "target": test_pred})
sub["id"] = sub["id"].astype(str)
sub["target"] = pd.to_numeric(sub["target"], errors="coerce").fillna(0.5).clip(0.0, 1.0)

sub.to_csv(out_path, index=False)
print(f"Wrote {out_path} with shape {sub.shape}")
print(sub.head())
