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

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone

np.random.seed(42)




## === cell 1
BASE_CANDIDATES = [
    "/kaggle/input/seti-breakthrough-listen",
    "/kaggle/data/seti-breakthrough-listen",
    "/kaggle/input",  # if competition dataset is mounted directly here
    "/kaggle/data",
]


def _first_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


base = _first_existing(*BASE_CANDIDATES)
if base is None:
    raise FileNotFoundError(
        "Could not locate Kaggle dataset directory under known /kaggle/* paths."
    )

if os.path.exists(os.path.join(base, "train")) and os.path.exists(
    os.path.join(base, "test")
):
    DATA_DIR = base
else:
    sub = _first_existing(os.path.join(base, "seti-breakthrough-listen"))
    if sub is None:
        raise FileNotFoundError(
            f"Could not locate train/test directories under: {base}"
        )
    DATA_DIR = sub

TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
LABELS_PATH = _first_existing(
    os.path.join(DATA_DIR, "train_labels.csv"),
    "/kaggle/input/train_labels.csv",
    "/kaggle/data/train_labels.csv",
)
SAMPLE_SUB_PATH = _first_existing(
    os.path.join(DATA_DIR, "sample_submission.csv"),
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
)

if LABELS_PATH is None:
    raise FileNotFoundError("train_labels.csv not found.")
if SAMPLE_SUB_PATH is None:
    raise FileNotFoundError("sample_submission.csv not found.")

train_labels = pd.read_csv(LABELS_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

for df, name in [(train_labels, "train_labels"), (sample_sub, "sample_submission")]:
    if not {"id", "target"}.issubset(df.columns):
        raise ValueError(
            f"{name} must contain columns ['id','target'], got: {df.columns.tolist()}"
        )

train_labels["id"] = train_labels["id"].astype(str)
sample_sub["id"] = sample_sub["id"].astype(str)

print("DATA_DIR:", DATA_DIR)
print("Train labels:", train_labels.shape, "Sample submission:", sample_sub.shape)




## === cell 2
def list_npy_files(root_dir):
    files = []
    with os.scandir(root_dir) as it:
        for entry in it:
            if entry.is_dir():
                with os.scandir(entry.path) as it2:
                    for f in it2:
                        if f.is_file() and f.name.endswith(".npy"):
                            files.append(f.path)
    return files


train_files = list_npy_files(TRAIN_DIR)
test_files = list_npy_files(TEST_DIR)

if len(train_files) == 0:
    raise FileNotFoundError(f"No train .npy files found under {TRAIN_DIR}")
if len(test_files) == 0:
    raise FileNotFoundError(f"No test .npy files found under {TEST_DIR}")


def id_from_path(p):
    return os.path.splitext(os.path.basename(p))[0]


train_path_by_id = {id_from_path(p): p for p in train_files}
test_path_by_id = {id_from_path(p): p for p in test_files}

train_id_set = set(train_path_by_id.keys())
test_id_set = set(test_path_by_id.keys())

missing_train = train_labels.loc[~train_labels["id"].isin(train_id_set), "id"]
missing_test = sample_sub.loc[~sample_sub["id"].isin(test_id_set), "id"]

if len(missing_train) > 0:
    raise FileNotFoundError(
        f"Missing {len(missing_train)} train ids in train folder. Example: {missing_train.iloc[:5].tolist()}"
    )
if len(missing_test) > 0:
    raise FileNotFoundError(
        f"Missing {len(missing_test)} test ids in test folder. Example: {missing_test.iloc[:5].tolist()}"
    )

print("Train npy files:", len(train_files), "Test npy files:", len(test_files))




## === cell 3
def extract_features_from_array(x):
    x = np.asarray(x, dtype=np.float32, order="C")

    A = x[[0, 2, 4]]  # (3,H,W)
    B = x[[1, 3, 5]]  # (3,H,W)

    A_mean = A.mean(axis=0)
    B_mean = B.mean(axis=0)
    diff = A_mean - B_mean
    absdiff = np.abs(diff)

    feats = np.empty(4 + 4 + 3 + 4 + 1 + 2, dtype=np.float32)
    k = 0

    feats[k] = A_mean.mean()
    k += 1
    feats[k] = B_mean.mean()
    k += 1
    feats[k] = diff.mean()
    k += 1
    feats[k] = absdiff.mean()
    k += 1

    feats[k] = A_mean.std()
    k += 1
    feats[k] = B_mean.std()
    k += 1
    feats[k] = diff.std()
    k += 1
    feats[k] = absdiff.std()
    k += 1

    ad_flat = absdiff.ravel()
    p = np.percentile(ad_flat, [90.0, 95.0, 99.0])
    feats[k : k + 3] = p.astype(np.float32, copy=False)
    k += 3

    row_mean = absdiff.mean(axis=1)  # (H,)
    col_mean = absdiff.mean(axis=0)  # (W,)
    feats[k] = row_mean.max()
    k += 1
    feats[k] = col_mean.max()
    k += 1
    feats[k] = row_mean.std()
    k += 1
    feats[k] = col_mean.std()
    k += 1

    eps = 1e-6
    feats[k] = (np.mean(A_mean * A_mean) + eps) / (np.mean(B_mean * B_mean) + eps)
    k += 1

    A_frame_means = A.reshape(3, -1).mean(axis=1)
    feats[k] = A_frame_means.std()
    k += 1
    B_frame_means = B.reshape(3, -1).mean(axis=1)
    feats[k] = B_frame_means.std()
    k += 1

    return feats


_tmp = np.load(train_path_by_id[train_labels["id"].iloc[0]])
_feat = extract_features_from_array(_tmp)
N_FEATS = int(_feat.shape[0])
assert N_FEATS > 0, "Feature dimension must be positive."
print("Feature dimension:", N_FEATS)


from concurrent.futures import ThreadPoolExecutor


def _load_one(path):
    arr = np.load(path)
    return extract_features_from_array(arr)


def load_and_extract(ids, path_by_id, max_workers=None, chunksize=256):
    paths = [path_by_id[_id] for _id in ids]
    X = np.empty((len(paths), N_FEATS), dtype=np.float32)

    if max_workers is None:
        cpu = os.cpu_count() or 4
        max_workers = min(8, cpu)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, feat in enumerate(ex.map(_load_one, paths, chunksize=chunksize)):
            X[i] = feat
    return X




## === cell 4
train_ids = train_labels["id"].tolist()
y = train_labels["target"].astype(np.int8).values

X = load_and_extract(train_ids, train_path_by_id)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof = np.zeros(len(train_ids), dtype=np.float32)

base_model = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        (
            "lr",
            LogisticRegression(
                solver="lbfgs",
                max_iter=2000,
                C=1.0,
                class_weight=None,
                n_jobs=None,
                random_state=42,
            ),
        ),
    ]
)

for fold, (tr, va) in enumerate(skf.split(X, y), 1):
    model = clone(base_model)
    model.fit(X[tr], y[tr])
    oof[va] = model.predict_proba(X[va])[:, 1].astype(np.float32)
    print(
        f"Fold {fold} done. Val preds: mean={oof[va].mean():.4f}, std={oof[va].std():.4f}"
    )

final_model = clone(base_model)
final_model.fit(X, y)




## === cell 5
test_ids = sample_sub["id"].tolist()
X_test = load_and_extract(test_ids, test_path_by_id)

test_pred = final_model.predict_proba(X_test)[:, 1].astype(np.float32)

submission = pd.DataFrame({"id": test_ids, "target": test_pred})
if submission["target"].isna().any():
    raise ValueError("NaNs detected in predictions.")
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
