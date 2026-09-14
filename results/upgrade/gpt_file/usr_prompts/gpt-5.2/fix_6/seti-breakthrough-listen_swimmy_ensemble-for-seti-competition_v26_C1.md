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
BASE_INPUT = "/kaggle/input"
DATA_ROOT = "/kaggle/data"

ROOT = BASE_INPUT if os.path.exists(BASE_INPUT) else DATA_ROOT

train_labels_path = os.path.join(ROOT, "train_labels.csv")
sample_sub_path = os.path.join(ROOT, "sample_submission.csv")
train_dir = os.path.join(ROOT, "train")
test_dir = os.path.join(ROOT, "test")

if not os.path.exists(train_labels_path):
    alt_root = os.path.join(ROOT, "seti-breakthrough-listen")
    if os.path.exists(os.path.join(alt_root, "train_labels.csv")):  # noqa: PTH110
        ROOT = alt_root
        train_labels_path = os.path.join(ROOT, "train_labels.csv")
        sample_sub_path = os.path.join(ROOT, "sample_submission.csv")
        train_dir = os.path.join(ROOT, "train")
        test_dir = os.path.join(ROOT, "test")

assert os.path.exists(
    sample_sub_path
), f"sample_submission.csv not found at {sample_sub_path}"
assert os.path.exists(
    train_labels_path
), f"train_labels.csv not found at {train_labels_path}"
assert os.path.isdir(train_dir), f"train dir not found at {train_dir}"
assert os.path.isdir(test_dir), f"test dir not found at {test_dir}"

sample_sub = pd.read_csv(sample_sub_path)
train_labels = pd.read_csv(train_labels_path)

print("ROOT:", ROOT)
print("sample_sub shape:", sample_sub.shape)
print("train_labels shape:", train_labels.shape)




## === cell 2
def list_npy_id_paths_sorted(base_dir):
    pairs = []
    with os.scandir(base_dir) as it:
        for entry in it:
            if not entry.is_dir():
                continue
            with os.scandir(entry.path) as it2:
                for f in it2:
                    name = f.name
                    if f.is_file() and name.endswith(".npy"):
                        _id = name[:-4]
                        pairs.append((_id, f.path))
    pairs.sort(key=lambda t: t[0])
    ids = [p[0] for p in pairs]
    paths = [p[1] for p in pairs]
    id2idx = {i: k for k, i in enumerate(ids)}
    return ids, paths, id2idx


train_ids, train_paths_list, train_id2idx = list_npy_id_paths_sorted(train_dir)
test_ids, test_paths_list, test_id2idx = list_npy_id_paths_sorted(test_dir)

print("Found train npy:", len(train_ids), "Found test npy:", len(test_ids))
print("Example train file:", train_paths_list[0] if train_paths_list else None)
print("Example test file:", test_paths_list[0] if test_paths_list else None)



## === cell 3
ensemble_candidates = [
    "../input/rerun-seti-e-t-volo-d1-baseline-inference/submission.csv",
    "../input/seti-bl-spatial-info-tf-tpu/submission.csv",
    "../input/seti-bl-tf-starter-tpu/submission.csv",
    "../input/seti-learned-image-resizing/submission.csv",
    "../input/lb-0-980-efficientnet-b0-more-epoch/submission.csv",
    "../input/inference-5x-ensemble-vanilla-resnet34d-seti/submission.csv",
]


def try_read_submission(path):
    if os.path.exists(path):
        df = pd.read_csv(path)
        if set(df.columns) >= {"id", "target"}:
            df = df[["id", "target"]].copy()
            return df
    return None


loaded = []
for p in ensemble_candidates:
    df = try_read_submission(p)
    loaded.append(df)

data1, data2, data3, data4, data5, data6 = loaded
print("Loaded ensemble files:", [d is not None for d in loaded])



## === cell 4
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

_N_TOTAL = 6 * 273 * 256
_P05_IDX = int(round(0.05 * (_N_TOTAL - 1)))
_P50_IDX = int(round(0.50 * (_N_TOTAL - 1)))
_P95_IDX = int(round(0.95 * (_N_TOTAL - 1)))


def extract_features_from_npy(arr):
    x = np.asarray(arr, dtype=np.float32)

    mean_all = x.mean()
    std_all = x.std()
    max_all = x.max()
    min_all = x.min()

    A = x[[0, 2, 4]]
    B = x[[1, 3, 5]]
    mean_A = A.mean()
    mean_B = B.mean()
    std_A = A.std()
    std_B = B.std()

    diff = A.mean(axis=0) - B.mean(axis=0)
    grad_t = np.mean(np.abs(np.diff(diff, axis=0)))
    grad_f = np.mean(np.abs(np.diff(diff, axis=1)))

    work = x.ravel().copy()
    np.partition(work, (_P05_IDX, _P50_IDX, _P95_IDX))
    p05 = work[_P05_IDX]
    p50 = work[_P50_IDX]
    p95 = work[_P95_IDX]

    contrast = p95 - p05
    skew_proxy = p95 + p05 - 2 * p50

    return np.array(
        [
            mean_all,
            std_all,
            max_all,
            min_all,
            mean_A,
            mean_B,
            std_A,
            std_B,
            grad_t,
            grad_f,
            contrast,
            skew_proxy,
        ],
        dtype=np.float32,
    )


_FEATURE_CACHE = {}


def _feat_worker(args):
    _id, path = args
    arr = np.load(path, mmap_mode="r")
    return _id, extract_features_from_npy(arr)


def build_feature_matrix_from_ids(ids, id2idx, paths_list, use_mp=True):
    n = len(ids)
    X = np.zeros((n, 12), dtype=np.float32)

    cache = _FEATURE_CACHE
    missing = []
    for _id in ids:
        if _id not in cache:
            missing.append(_id)

    if missing:
        jobs = [(_id, paths_list[id2idx[_id]]) for _id in missing]
        if use_mp and len(jobs) >= 256:
            import multiprocessing as mp

            ctx = mp.get_context("fork")
            nproc = min(8, os.cpu_count() or 2)
            with ctx.Pool(processes=nproc, maxtasksperchild=512) as pool:
                for _id, feat in pool.imap_unordered(_feat_worker, jobs, chunksize=64):
                    cache[_id] = feat
        else:
            np_load = np.load
            for _id, path in jobs:
                arr = np_load(path, mmap_mode="r")
                cache[_id] = extract_features_from_npy(arr)

    for i, _id in enumerate(ids):
        X[i] = cache[_id]
    return X




## === cell 5
need_fallback = not (data1 is not None and data4 is not None and data5 is not None)

baseline_sub = None
baseline_oof_auc = None

if need_fallback:
    train_df = train_labels[train_labels["id"].isin(train_id2idx)].copy()
    train_df = train_df.sort_values("id").reset_index(drop=True)

    y = train_df["target"].values.astype(int, copy=False)
    X = build_feature_matrix_from_ids(
        train_df["id"].tolist(), train_id2idx, train_paths_list, use_mp=True
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(train_df), dtype=np.float32)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="lbfgs", max_iter=200, n_jobs=1, C=1.0, class_weight=None
                ),
            ),
        ]
    )

    for tr_idx, va_idx in skf.split(X, y):
        model.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = model.predict_proba(X[va_idx])[:, 1].astype(np.float32)

    try:
        from sklearn.metrics import roc_auc_score

        baseline_oof_auc = roc_auc_score(y, oof)
        print("Fallback baseline CV AUC:", baseline_oof_auc)
    except Exception as e:
        print("Could not compute CV AUC:", repr(e))

    model.fit(X, y)

    test_ids_ordered = sample_sub["id"].tolist()
    missing = [i for i in test_ids_ordered if i not in test_id2idx]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} test npy files referenced by sample_submission. Example: {missing[0]}"
        )

    X_test = build_feature_matrix_from_ids(
        test_ids_ordered, test_id2idx, test_paths_list, use_mp=True
    )
    test_pred = model.predict_proba(X_test)[:, 1].astype(np.float32)

    baseline_sub = pd.DataFrame({"id": test_ids_ordered, "target": test_pred})
    print("Built fallback submission from local model. Shape:", baseline_sub.shape)




## === cell 6
def align_to_sample(df, sample_ids):
    df = df[["id", "target"]]
    df = df.set_index("id").reindex(sample_ids)
    if df["target"].isna().any():
        missing = int(df["target"].isna().sum())
        raise ValueError(
            f"Submission missing {missing} ids after alignment to sample_submission."
        )
    return df.reset_index()


sample_ids = sample_sub["id"].tolist()

final_sub = None
if (data1 is not None) and (data4 is not None) and (data5 is not None):
    d1 = align_to_sample(data1, sample_ids)
    d4 = align_to_sample(data4, sample_ids)
    d5 = align_to_sample(data5, sample_ids)

    final = d1.copy()
    final["target"] = (
        0.76 * d5["target"].astype(float).to_numpy()
        + 0.14 * d4["target"].astype(float).to_numpy()
        + 0.10 * d1["target"].astype(float).to_numpy()
    )
    final["target"] = final["target"].clip(0.0, 1.0)
    final_sub = final
    print("Created blended submission using available external files.")
else:
    if baseline_sub is None:
        final_sub = sample_sub.copy()
        final_sub["target"] = 0.5
        print(
            "WARNING: No ensemble inputs and baseline failed; writing 0.5 predictions."
        )
    else:
        final_sub = baseline_sub.copy()
        print(
            "Using fallback baseline submission (external ensemble files not available)."
        )

final_sub = final_sub[["id", "target"]].copy()
assert final_sub.shape[0] == sample_sub.shape[0], "Submission row count mismatch."
assert (
    final_sub["id"].tolist() == sample_sub["id"].tolist()
), "Submission id order mismatch."

final_sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", final_sub.shape)
print(final_sub.head())
