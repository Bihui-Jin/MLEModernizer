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

_CPU = os.cpu_count() or 1
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

from sklearn.model_selection import (
    train_test_split,
)  # noqa: F401 (kept to preserve original structure)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

BASE_DIR = "/kaggle/input"
CANDIDATE_BASES = [
    os.path.join(BASE_DIR, "seti-breakthrough-listen"),
    BASE_DIR,
]
DATA_BASE = None
for cand in CANDIDATE_BASES:
    if os.path.isdir(os.path.join(cand, "train")) and os.path.isdir(
        os.path.join(cand, "test")
    ):
        DATA_BASE = cand
        break
if DATA_BASE is None:
    raise FileNotFoundError(
        "Could not locate dataset base dir containing train/ and test/ under /kaggle/input."
    )

TRAIN_DIR = os.path.join(DATA_BASE, "train")
TEST_DIR = os.path.join(DATA_BASE, "test")
TRAIN_LABELS_PATH = os.path.join(DATA_BASE, "train_labels.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_BASE, "sample_submission.csv")

assert os.path.exists(TRAIN_LABELS_PATH), f"Missing: {TRAIN_LABELS_PATH}"
assert os.path.exists(SAMPLE_SUB_PATH), f"Missing: {SAMPLE_SUB_PATH}"
assert os.path.isdir(TRAIN_DIR), f"Missing dir: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing dir: {TEST_DIR}"

train_labels = pd.read_csv(TRAIN_LABELS_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

print("Using DATA_BASE:", DATA_BASE)
print("train_labels shape:", train_labels.shape)
print("sample_sub shape:", sample_sub.shape)




## === cell 1
from concurrent.futures import ThreadPoolExecutor


def extract_features_from_npy(path: str) -> np.ndarray:
    x = np.load(path, mmap_mode="r")  # float16 on disk; mmap reduces IO overhead
    x = x.astype(np.float32, copy=False)

    absx = np.abs(x)

    panel_mean = x.mean(axis=(1, 2))  # (6,)
    panel_std = x.std(axis=(1, 2))  # (6,)
    panel_absmean = absx.mean(axis=(1, 2))
    panel_absmax = absx.max(axis=(1, 2))

    A = x[[0, 2, 4]]
    BCD = x[[1, 3, 5]]

    A_mean = A.mean()
    BCD_mean = BCD.mean()
    A_std = A.std()
    BCD_std = BCD.std()

    mean_diff = A_mean - BCD_mean
    std_diff = A_std - BCD_std

    A_panel_mean = panel_mean[[0, 2, 4]]
    BCD_panel_mean = panel_mean[[1, 3, 5]]
    A_panel_std = panel_std[[0, 2, 4]]
    BCD_panel_std = panel_std[[1, 3, 5]]

    feats_tail = np.array(
        [
            A_mean,
            BCD_mean,
            mean_diff,
            A_std,
            BCD_std,
            std_diff,
            A_panel_mean.mean(),
            BCD_panel_mean.mean(),
            (A_panel_mean - BCD_panel_mean).mean(),
            A_panel_std.mean(),
            BCD_panel_std.mean(),
            (A_panel_std - BCD_panel_std).mean(),
            A_panel_mean.std(),
            BCD_panel_mean.std(),
            A_panel_std.std(),
            BCD_panel_std.std(),
        ],
        dtype=np.float32,
    )

    feats = np.concatenate(
        [panel_mean, panel_std, panel_absmean, panel_absmax, feats_tail]
    ).astype(np.float32, copy=False)

    return feats


def build_id_to_path_map(root_dir: str) -> dict:
    id2p = {}
    with os.scandir(root_dir) as it:
        for entry in it:
            if not entry.is_dir():
                continue
            subdir = entry.path
            with os.scandir(subdir) as it2:
                for f in it2:
                    if not f.is_file():
                        continue
                    name = f.name
                    if name.endswith(".npy"):
                        id2p[name[:-4]] = f.path
    return id2p


def extract_features_for_ids(
    ids: np.ndarray, id2path: dict, n_feats: int, max_workers: int = None
) -> np.ndarray:
    n = len(ids)
    X_out = np.empty((n, n_feats), dtype=np.float32)

    if max_workers is None:
        max_workers = min(16, os.cpu_count() or 4)

    _id2path = id2path
    _extract = extract_features_from_npy

    def _feat_from_id(_id: str) -> np.ndarray:
        return _extract(_id2path[_id])

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, feats in enumerate(ex.map(_feat_from_id, ids, chunksize=64)):
            X_out[i] = feats
    return X_out




## === cell 2
train_id_to_path = build_id_to_path_map(TRAIN_DIR)

train_ids = train_labels["id"].to_numpy()
train_targets = train_labels["target"].to_numpy(dtype=np.int64, copy=False)

idset = set(train_id_to_path.keys())
missing = [str(_id) for _id in train_ids if _id not in idset]
if missing:
    raise FileNotFoundError(
        f"Missing {len(missing)} training npy files, e.g. {missing[:5]}"
    )

first_feat = extract_features_from_npy(train_id_to_path[train_ids[0]])
n_feats = int(first_feat.shape[0])

X = extract_features_for_ids(train_ids, train_id_to_path, n_feats)
y = train_targets

print("Train X/y shapes:", X.shape, y.shape)




## === cell 3
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import itertools

C_CANDIDATES = np.logspace(-6, 3, 10).astype(float).tolist()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_auc = -1.0
best_params = None

candidates = [
    ("l2", "lbfgs"),
    ("l2", "liblinear"),
    ("l1", "liblinear"),
]

fold_data = []
for tr_idx, va_idx in skf.split(X, y):
    X_tr = np.ascontiguousarray(X[tr_idx], dtype=np.float32)
    X_va = np.ascontiguousarray(X[va_idx], dtype=np.float32)

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)

    fold_data.append((tr_idx, va_idx, X_tr_s, y[tr_idx], X_va_s))


def oof_auc_for_params_precomputed(y_full, penalty, solver, C, fold_data):
    oof = np.empty(y_full.shape[0], dtype=np.float32)
    lr = LogisticRegression  # local bind

    for tr_idx, va_idx, X_tr_s, y_tr, X_va_s in fold_data:
        clf = lr(
            solver=solver,
            penalty=penalty,
            max_iter=2000,
            n_jobs=1,  # avoid nested parallelism; semantics identical
            class_weight=None,
            random_state=42,
            C=C,
        )
        clf.fit(X_tr_s, y_tr)
        oof[va_idx] = clf.predict_proba(X_va_s)[:, 1]
    return roc_auc_score(y_full, oof)


param_grid = [(p, s, c) for (p, s), c in itertools.product(candidates, C_CANDIDATES)]

results = []
for penalty, solver, C in param_grid:
    auc = oof_auc_for_params_precomputed(y, penalty, solver, C, fold_data)
    results.append((penalty, solver, C, float(auc)))

for penalty, solver, C, auc in sorted(results, key=lambda t: (t[0], t[1], t[2])):
    print(f"penalty={penalty:<2} solver={solver:<9} C={C:<10.4g}  OOF AUC={auc:.6f}")
    if auc > best_auc:
        best_auc = auc
        best_params = (penalty, solver, C)

best_penalty, best_solver, best_C = best_params
print(
    f"Selected penalty={best_penalty}, solver={best_solver}, C={best_C} with OOF AUC={best_auc:.6f}"
)

final_scaler = StandardScaler(with_mean=True, with_std=True)
X_s = final_scaler.fit_transform(X)

final_clf = LogisticRegression(
    solver=best_solver,
    penalty=best_penalty,
    max_iter=2000,
    n_jobs=1,
    class_weight=None,
    random_state=42,
    C=best_C,
)
final_clf.fit(X_s, y)




## === cell 4
test_id_to_path = build_id_to_path_map(TEST_DIR)

test_ids = sample_sub["id"].to_numpy()

test_idset = set(test_id_to_path.keys())
missing_test = [str(_id) for _id in test_ids if _id not in test_idset]
if missing_test:
    raise FileNotFoundError(
        f"Missing {len(missing_test)} test npy files, e.g. {missing_test[:5]}"
    )

X_test = extract_features_for_ids(test_ids, test_id_to_path, n_feats)
X_test_s = final_scaler.transform(X_test)

test_pred = final_clf.predict_proba(X_test_s)[:, 1].astype(np.float32, copy=False)

submission = pd.DataFrame({"id": test_ids, "target": test_pred})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("submission.csv path:", os.path.abspath("submission.csv"))
