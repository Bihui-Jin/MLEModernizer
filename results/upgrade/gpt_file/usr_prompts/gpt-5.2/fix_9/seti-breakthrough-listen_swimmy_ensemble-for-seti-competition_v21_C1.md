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

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

np.random.seed(42)



## === cell 1
BASE_DIR = "/kaggle/data"
TRAIN_LABELS_PATH = os.path.join(BASE_DIR, "train_labels.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_DIR, "sample_submission.csv")
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

assert os.path.exists(TRAIN_LABELS_PATH), f"Missing: {TRAIN_LABELS_PATH}"
assert os.path.exists(SAMPLE_SUB_PATH), f"Missing: {SAMPLE_SUB_PATH}"
assert os.path.isdir(TRAIN_DIR), f"Missing dir: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing dir: {TEST_DIR}"

train_labels = pd.read_csv(TRAIN_LABELS_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

assert set(train_labels.columns) >= {"id", "target"}
assert list(sample_sub.columns)[:2] == ["id", "target"]
train_labels["id"] = train_labels["id"].astype(str)
sample_sub["id"] = sample_sub["id"].astype(str)

print("train_labels:", train_labels.shape, "sample_submission:", sample_sub.shape)




## === cell 2
def build_id_to_path_map(root_dir: str):
    id_to_path = {}
    with os.scandir(root_dir) as it:
        for entry in it:
            if not entry.is_dir():
                continue
            subdir = entry.path
            with os.scandir(subdir) as it2:
                for f in it2:
                    if f.is_file() and f.name.endswith(".npy"):
                        fid = f.name[:-4]  # strip ".npy"
                        id_to_path[fid] = f.path
    return id_to_path


train_id_to_path = build_id_to_path_map(TRAIN_DIR)
test_id_to_path = build_id_to_path_map(TEST_DIR)

print(
    "Found train npy:", len(train_id_to_path), "Found test npy:", len(test_id_to_path)
)

missing_train = train_labels.loc[~train_labels["id"].isin(train_id_to_path)].shape[0]
print("Missing train files for labels:", missing_train)
assert missing_train == 0, "Some train label ids do not have corresponding .npy files."




## === cell 3
def extract_features_from_array(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    feats = []

    panel_mean = x.mean(axis=(1, 2))
    panel_std = x.std(axis=(1, 2))
    panel_max = x.max(axis=(1, 2))
    panel_p95 = np.percentile(x, 95, axis=(1, 2))

    feats.extend(
        np.stack([panel_mean, panel_std, panel_max, panel_p95], axis=1).ravel().tolist()
    )

    A = x[[0, 2, 4]].mean(axis=0)
    B = x[[1, 3, 5]].mean(axis=0)
    D = A - B

    feats.append(A.mean())
    feats.append(A.std())
    feats.append(B.mean())
    feats.append(B.std())
    feats.append(D.mean())
    feats.append(D.std())
    feats.append(D.max())
    feats.append(np.percentile(D, 95))

    grad_t = np.abs(np.diff(D, axis=0)).mean()
    grad_f = np.abs(np.diff(D, axis=1)).mean()
    feats.append(grad_t)
    feats.append(grad_f)

    d2 = D * D
    feats.append(d2.mean())  # energy in D
    feats.append(np.sqrt(d2.mean() + 1e-8))  # RMS in D
    feats.append(np.percentile(np.abs(D), 99))  # extreme contrast

    row_prof = D.mean(axis=1)  # (273,)
    col_prof = D.mean(axis=0)  # (256,)

    feats.append(row_prof.std())
    feats.append(col_prof.std())
    feats.append(np.max(row_prof) - np.min(row_prof))
    feats.append(np.max(col_prof) - np.min(col_prof))

    a_flat = A.ravel()
    b_flat = B.ravel()
    denom = (np.linalg.norm(a_flat) * np.linalg.norm(b_flat)) + 1e-8
    feats.append(float(np.dot(a_flat, b_flat) / denom))

    pos_mean = float(np.maximum(D, 0).mean())
    neg_mean = float(np.maximum(-D, 0).mean())
    feats.append(pos_mean)
    feats.append(neg_mean)

    D_abs = np.abs(D)
    feats.append(float(D_abs.mean()))
    feats.append(float(D_abs.std()))
    feats.append(float(np.percentile(D_abs, 90)))
    feats.append(float(np.percentile(D_abs, 95)))
    feats.append(float(np.percentile(D_abs, 99)))

    mu = float(D.mean())
    sigma = float(D.std() + 1e-6)
    z = (D - mu) / sigma
    feats.append(float((z**3).mean()))  # skewness approx
    feats.append(float((z**4).mean()))  # kurtosis approx (not excess)

    feats.append(float(D_abs.sum(axis=0).max()))  # max over frequency columns
    feats.append(float(D_abs.sum(axis=1).max()))  # max over time rows

    feats.append(float(np.linalg.norm(D.ravel()) / (np.linalg.norm(B.ravel()) + 1e-6)))

    return np.asarray(feats, dtype=np.float32)


from concurrent.futures import ThreadPoolExecutor

N_FEATS = int(
    extract_features_from_array(np.zeros((6, 273, 256), dtype=np.float32)).shape[0]
)
print("N_FEATS =", N_FEATS)


def _featurize_one(args):
    i, path = args
    arr = np.load(path)
    feat = extract_features_from_array(arr)
    return i, feat


def load_and_featurize(ids, id_to_path_map, batch_size=8192, max_workers=None):
    n = len(ids)
    X_out = np.empty((n, N_FEATS), dtype=np.float32)

    if max_workers is None:
        cpu = os.cpu_count() or 4
        max_workers = min(16, cpu)

    paths = [id_to_path_map[fid] for fid in ids]

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            tasks = [(i, paths[i]) for i in range(start, end)]
            for i, feat in ex.map(_featurize_one, tasks, chunksize=64):
                if feat.shape[0] != N_FEATS:
                    raise ValueError(
                        f"Feature length mismatch for index={i}: got {feat.shape[0]}, expected {N_FEATS}"
                    )
                X_out[i] = feat

    return X_out


train_ids = train_labels["id"].tolist()
y = train_labels["target"].astype(np.int8).values

print("Featurizing train...")
X = load_and_featurize(train_ids, train_id_to_path, batch_size=8192)
print("X shape:", X.shape, "y shape:", y.shape)
assert X.shape[1] == N_FEATS



## === cell 4
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

C_GRID = [0.3, 1.0, 3.0, 10.0]
CW_GRID = [None, "balanced"]


splits = list(skf.split(X, y))

scaler = StandardScaler(with_mean=True, with_std=True)
X_scaled = scaler.fit_transform(X).astype(np.float32, copy=False)

_LR_NJOBS = max(1, (os.cpu_count() or 4))


def make_lr(C=1.0, class_weight=None):
    return LogisticRegression(
        max_iter=400,
        solver="lbfgs",
        n_jobs=_LR_NJOBS,
        C=C,
        class_weight=class_weight,
    )


best_params = None
best_cv_auc = -1.0

print("Selecting C/class_weight via CV (same folds)...")
for C in C_GRID:
    for cw in CW_GRID:
        oof_tmp = np.zeros(len(y), dtype=np.float32)
        lr_tmp = make_lr(C=C, class_weight=cw)
        for fold, (tr_idx, va_idx) in enumerate(splits, 1):
            lr_tmp = make_lr(C=C, class_weight=cw)
            lr_tmp.fit(X_scaled[tr_idx], y[tr_idx])
            oof_tmp[va_idx] = lr_tmp.predict_proba(X_scaled[va_idx])[:, 1].astype(
                np.float32
            )
        cv_auc_tmp = roc_auc_score(y, oof_tmp)
        print(f"  C={C:<4} class_weight={str(cw):<8} CV AUC={cv_auc_tmp:.6f}")
        if cv_auc_tmp > best_cv_auc:
            best_cv_auc = cv_auc_tmp
            best_params = (C, cw)

print(
    "Best params:",
    {"C": best_params[0], "class_weight": best_params[1]},
    "CV AUC:",
    f"{best_cv_auc:.6f}",
)

C_BEST, CW_BEST = best_params

oof = np.zeros(len(y), dtype=np.float32)
print("Cross-validating with best params...")
for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    lr = make_lr(C=C_BEST, class_weight=CW_BEST)
    lr.fit(X_scaled[tr_idx], y[tr_idx])
    oof[va_idx] = lr.predict_proba(X_scaled[va_idx])[:, 1].astype(np.float32)
    fold_auc = roc_auc_score(y[va_idx], oof[va_idx])
    print(f"Fold {fold} AUC: {fold_auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"Overall CV AUC: {cv_auc:.6f}")

print("Preparing fold-ensemble (we will still also fit on full data for reference)...")
print("Fitting on full training data...")
lr_full = make_lr(C=C_BEST, class_weight=CW_BEST)
lr_full.fit(X_scaled, y)



## === cell 5
test_ids = sample_sub["id"].tolist()

missing_test = [fid for fid in test_ids if fid not in test_id_to_path]
assert (
    len(missing_test) == 0
), f"Missing {len(missing_test)} test files. Example: {missing_test[:3]}"

print("Featurizing test...")
X_test = load_and_featurize(test_ids, test_id_to_path, batch_size=8192)
print("X_test shape:", X_test.shape)
assert X_test.shape[1] == N_FEATS

X_test_scaled = scaler.transform(X_test).astype(np.float32, copy=False)

test_pred_cv = np.zeros(X_test_scaled.shape[0], dtype=np.float32)
for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    lr_fold = make_lr(C=C_BEST, class_weight=CW_BEST)
    lr_fold.fit(X_scaled[tr_idx], y[tr_idx])
    test_pred_cv += lr_fold.predict_proba(X_test_scaled)[:, 1].astype(np.float32)
test_pred_cv /= len(splits)

test_pred_full = lr_full.predict_proba(X_test_scaled)[:, 1].astype(np.float32)

test_pred = np.clip(test_pred_cv, 0.0, 1.0)



## === cell 6
submission = pd.DataFrame({"id": test_ids, "target": test_pred})
assert submission.shape[0] == sample_sub.shape[0]
assert (submission["id"].values == sample_sub["id"].values).all()

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission.head())
