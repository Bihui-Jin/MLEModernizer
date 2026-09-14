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
CANDIDATE_SUB_PATHS = [
    "../input/rerun-seti-e-t-volo-d1-baseline-inference/submission.csv",
    "../input/lb-0-980-efficientnet-b0-more-epoch/submission.csv",
    "../input/inference-5x-ensemble-vanilla-resnet34d-seti/submission.csv",
    "../input/seti-learned-image-resizing/submission.csv",
    "../input/rerun-seti-e-t-resnet18d-baseline/submission.csv",
    "../input/ensemble-for-seti-competition/submission.csv",
    "../input/fixed-gradual-warmup-custom-head/submission.csv",
]

SAMPLE_SUB_PATHS = [
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
    "../input/sample_submission.csv",
]
sample_path = next((p for p in SAMPLE_SUB_PATHS if os.path.exists(p)), None)
if sample_path is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv in expected locations: "
        + ", ".join(SAMPLE_SUB_PATHS)
    )

sample = pd.read_csv(sample_path)
if not {"id", "target"}.issubset(sample.columns):
    raise ValueError(
        f"sample_submission.csv must contain columns ['id','target'], got: {list(sample.columns)}"
    )


def _load_submission_if_exists(path: str, sample_ids: pd.Series) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if not {"id", "target"}.issubset(df.columns):
        return None
    df = df[["id", "target"]].drop_duplicates("id")
    df = sample_ids.to_frame(index=False).merge(df, on="id", how="left")
    return df


sample_ids = sample["id"]
loaded = [_load_submission_if_exists(p, sample_ids) for p in CANDIDATE_SUB_PATHS]

data1, data2, data3, data4, data5, data6, data7 = loaded

available = [i + 1 for i, d in enumerate(loaded) if d is not None]
print(f"Loaded {len(available)} external submission(s): {available} (out of 7).")
print(f"Using sample_submission.csv from: {sample_path}")



## === cell 2
data11 = data1.copy() if data1 is not None else sample[["id", "target"]].copy()



## === cell 3
weights = {
    "data1": 0.12,
    "data2": 0.10,
    "data3": 0.10,
    "data4": 0.11,
    "data5": 0.11,
    "data6": 0.58,
}
sources = {
    "data1": data1,
    "data2": data2,
    "data3": data3,
    "data4": data4,
    "data5": data5,
    "data6": data6,
}

present = [(name, df, weights[name]) for name, df in sources.items() if df is not None]
if len(present) == 0:
    data11["target"] = np.nan
else:
    wsum = sum(w for _, _, w in present)
    preds = np.zeros(len(sample_ids), dtype=np.float64)
    for name, df, w in present:
        p = (
            pd.to_numeric(df["target"], errors="coerce")
            .fillna(0.5)
            .to_numpy(dtype=np.float64)
        )
        preds += (w / wsum) * p

    data11["target"] = np.clip(preds, 0.0, 1.0)

data11 = sample[["id"]].merge(data11[["id", "target"]], on="id", how="left")




## === cell 4
def _find_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


TRAIN_LABELS_PATHS = [
    "/kaggle/input/train_labels.csv",
    "/kaggle/data/train_labels.csv",
    "../input/train_labels.csv",
]
train_labels_path = _find_first_existing(TRAIN_LABELS_PATHS)

TRAIN_DIR_CANDIDATES = [
    "/kaggle/input/train",
    "/kaggle/data/train",
    "../input/train",
]
TEST_DIR_CANDIDATES = [
    "/kaggle/input/test",
    "/kaggle/data/test",
    "../input/test",
]
train_dir = _find_first_existing(TRAIN_DIR_CANDIDATES)
test_dir = _find_first_existing(TEST_DIR_CANDIDATES)


def _index_npy_paths(base_dir: str) -> dict[str, str]:
    paths = glob.glob(os.path.join(base_dir, "*", "*.npy"))
    return {os.path.splitext(os.path.basename(p))[0]: p for p in paths}


def _quantile_linear_1d(a: np.ndarray, q: float) -> float:
    n = a.size
    if n == 0:
        return float("nan")
    if q <= 0.0:
        return float(np.min(a))
    if q >= 1.0:
        return float(np.max(a))
    h = (n - 1) * q
    i = int(np.floor(h))
    frac = h - i
    if frac == 0.0:
        return float(np.partition(a, i)[i])
    ai = np.partition(a, i)[i]
    aj = np.partition(a, i + 1)[i + 1]
    return float(ai + (aj - ai) * frac)


def _extract_features_from_array(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    a = x[[0, 2, 4]]
    b = x[[1, 3, 5]]

    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    d_flat = (a - b).reshape(-1)

    def stats_flat(zf: np.ndarray) -> np.ndarray:
        m = float(zf.mean())
        s = float(zf.std())
        q10 = _quantile_linear_1d(zf, 0.10)
        q50 = _quantile_linear_1d(zf, 0.50)
        q90 = _quantile_linear_1d(zf, 0.90)
        return np.array([m, s, q10, q50, q90], dtype=np.float32)

    fa = stats_flat(a_flat)
    fb = stats_flat(b_flat)
    fd = stats_flat(d_flat)

    amax = float(a_flat.max())
    bmax = float(b_flat.max())
    amin = float(a_flat.min())
    bmin = float(b_flat.min())

    aq99 = _quantile_linear_1d(a_flat, 0.99)
    bq99 = _quantile_linear_1d(b_flat, 0.99)

    extra = np.array(
        [
            amax,
            bmax,
            amax - amin,
            bmax - bmin,
            float((a_flat > aq99).mean()),
            float((b_flat > bq99).mean()),
        ],
        dtype=np.float32,
    )
    return np.concatenate([fa, fb, fd, extra], axis=0)


def _build_feature_matrix(
    ids: list[str], id_to_path: dict[str, str], max_items: int | None = None
) -> tuple[np.ndarray, list[str]]:
    n_in = len(ids) if max_items is None else min(len(ids), max_items)
    feat_dim = 21  # 3*5 stats + 6 extra
    X = np.empty((n_in, feat_dim), dtype=np.float32)
    kept_ids: list[str] = []
    out_i = 0
    for i, id_ in enumerate(ids[:n_in]):
        p = id_to_path.get(id_)
        if p is None:
            continue
        arr = np.load(p)  # float16 on disk
        X[out_i] = _extract_features_from_array(arr)
        kept_ids.append(id_)
        out_i += 1
        if (i + 1) % 4000 == 0:
            print(f"  processed {i+1}/{n_in}")
    if out_i == 0:
        return np.zeros((0, feat_dim), dtype=np.float32), []
    return X[:out_i], kept_ids


if data11["target"].isna().any():
    if train_labels_path is None or train_dir is None or test_dir is None:
        print(
            "Fallback training disabled (missing train/test paths). Writing 0.5 predictions."
        )
        data11["target"] = 0.5
    else:
        from sklearn.model_selection import StratifiedKFold
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        print(f"Indexing train npy paths under: {train_dir}")
        train_id_to_path = _index_npy_paths(train_dir)
        print(f"Indexing test npy paths under: {test_dir}")
        test_id_to_path = _index_npy_paths(test_dir)

        train_labels = pd.read_csv(train_labels_path)
        train_labels = train_labels[["id", "target"]].drop_duplicates("id")

        train_ids = train_labels["id"].tolist()
        print(f"Building train features for {len(train_ids)} ids from: {train_dir}")
        X_train, kept_train_ids = _build_feature_matrix(
            train_ids, train_id_to_path, max_items=None
        )
        y_train = (
            train_labels.set_index("id")
            .loc[kept_train_ids, "target"]
            .to_numpy(dtype=np.int64)
        )
        print(f"Train feature matrix: {X_train.shape}")

        test_ids = sample["id"].tolist()
        print(f"Building test features for {len(test_ids)} ids from: {test_dir}")
        X_test, kept_test_ids = _build_feature_matrix(
            test_ids, test_id_to_path, max_items=None
        )
        print(f"Test feature matrix: {X_test.shape}")

        kept_test_idx = pd.Index(kept_test_ids)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof = np.zeros(len(kept_train_ids), dtype=np.float64)
        test_pred = np.zeros(len(kept_test_ids), dtype=np.float64)

        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=300, solver="lbfgs")),
            ]
        )

        for fold, (tr, va) in enumerate(skf.split(X_train, y_train), 1):
            model.fit(X_train[tr], y_train[tr])
            oof[va] = model.predict_proba(X_train[va])[:, 1]
            test_pred += model.predict_proba(X_test)[:, 1] / skf.n_splits
            print(f"Fold {fold} done.")

        pred_map = pd.Series(test_pred, index=kept_test_idx)
        data11["target"] = data11["id"].map(pred_map).astype("float64")
        data11["target"] = data11["target"].fillna(0.5).clip(0.0, 1.0)



## === cell 5
out_path = "submission.csv"
data11[["id", "target"]].to_csv(out_path, index=False)

check = pd.read_csv(out_path)
assert list(check.columns) == [
    "id",
    "target",
], f"Bad submission columns: {list(check.columns)}"
assert len(check) == len(
    sample
), f"Row count mismatch vs sample: {len(check)} vs {len(sample)}"
print(f"Wrote {out_path} with shape {check.shape}")
print(check.head())
