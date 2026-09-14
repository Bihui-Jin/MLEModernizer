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
np.random.seed(42)



## === cell 1
candidate_paths = [
    "/kaggle/input/rerun-seti-e-t-volo-d1-baseline-inference/submission.csv",
    "/kaggle/input/lb-0-980-efficientnet-b0-more-epoch/submission.csv",
    "/kaggle/input/inference-5x-ensemble-vanilla-resnet34d-seti/submission.csv",
    "/kaggle/input/seti-learned-image-resizing/submission.csv",
    "/kaggle/input/rerun-seti-e-t-resnet18d-baseline/submission.csv",
    "/kaggle/input/ensemble-for-seti-competition/submission.csv",
    "/kaggle/input/fixed-gradual-warmup-custom-head/submission.csv",
]

loaded = []
loaded_paths = []
for p in candidate_paths:
    if os.path.exists(p):
        df = pd.read_csv(p, usecols=["id", "target"])
        loaded.append(df.copy())
        loaded_paths.append(p)

print(f"Found {len(loaded)} external submission files.")
if loaded_paths:
    print("Loaded:", loaded_paths[:3], "..." if len(loaded_paths) > 3 else "")



## === cell 2
submission = None

if len(loaded) > 0:
    base = loaded[0].sort_values("id").reset_index(drop=True)
    base_ids = base["id"].to_numpy()

    preds = np.empty((len(base_ids), len(loaded)), dtype=np.float64)
    for j, df in enumerate(loaded):
        if df.shape[0] != base.shape[0] or not df["id"].is_monotonic_increasing:
            df2 = df.sort_values("id").reset_index(drop=True)
        else:
            df2 = df
        if not np.array_equal(df2["id"].to_numpy(), base_ids):
            aligned = df2.set_index("id").reindex(base_ids)["target"].to_numpy()
            preds[:, j] = aligned.astype(np.float64, copy=False)
        else:
            preds[:, j] = df2["target"].to_numpy(dtype=np.float64, copy=False)

    orig_w = np.array([0.11, 0.11, 0.10, 0.11, 0.10, 0.52, 0.0], dtype=np.float64)
    w = orig_w[: preds.shape[1]]
    if w.sum() == 0:
        w = np.ones(preds.shape[1], dtype=np.float64)
    w = w / w.sum()

    ens = preds @ w
    submission = pd.DataFrame({"id": base_ids, "target": np.clip(ens, 0.0, 1.0)})



## === cell 3
if submission is None:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    import multiprocessing as mp

    CANDIDATE_ROOTS = [
        "/kaggle/input/seti-breakthrough-listen",
        "/kaggle/data/seti-breakthrough-listen",
        "/kaggle/data",
        "/kaggle/input",
    ]

    def resolve_comp_root():
        for r in CANDIDATE_ROOTS:
            if not os.path.exists(r):
                continue
            if (
                os.path.exists(os.path.join(r, "train_labels.csv"))
                and os.path.exists(os.path.join(r, "sample_submission.csv"))
                and os.path.isdir(os.path.join(r, "train"))
                and os.path.isdir(os.path.join(r, "test"))
            ):
                return r
            sub = os.path.join(r, "seti-breakthrough-listen")
            if (
                os.path.exists(os.path.join(sub, "train_labels.csv"))
                and os.path.exists(os.path.join(sub, "sample_submission.csv"))
                and os.path.isdir(os.path.join(sub, "train"))
                and os.path.isdir(os.path.join(sub, "test"))
            ):
                return sub
        return None

    root = resolve_comp_root()
    if root is None:
        raise FileNotFoundError(
            "Could not find competition data root. Looked for train_labels.csv/sample_submission.csv "
            "and train/test folders under: " + ", ".join(CANDIDATE_ROOTS)
        )

    train_labels_path = os.path.join(root, "train_labels.csv")
    sample_sub_path = os.path.join(root, "sample_submission.csv")
    train_dir = os.path.join(root, "train")
    test_dir = os.path.join(root, "test")

    labels = pd.read_csv(train_labels_path, usecols=["id", "target"])
    sample_sub = pd.read_csv(sample_sub_path, usecols=["id"])

    def id_to_path(base_dir: str, _id: str) -> str:
        return os.path.join(base_dir, _id[0], f"{_id}.npy")

    def paths_for_ids(base_dir: str, ids: np.ndarray):
        return [id_to_path(base_dir, _id) for _id in ids]

    train_ids_all = labels["id"].to_numpy()
    train_paths_all = paths_for_ids(train_dir, train_ids_all)
    exists_mask = np.fromiter(
        (os.path.exists(p) for p in train_paths_all),
        dtype=np.bool_,
        count=len(train_paths_all),
    )
    labels = labels.iloc[np.flatnonzero(exists_mask)].reset_index(drop=True)

    def extract_features_from_path(path):
        x = np.load(path)  # float16
        x = x.astype(np.float32, copy=False)

        means = x.mean(axis=(1, 2))
        stds = x.std(axis=(1, 2))
        maxs = x.max(axis=(1, 2))

        A = x[[0, 2, 4]]
        BCD = x[[1, 3, 5]]

        a_mean = A.mean()
        b_mean = BCD.mean()
        a_std = A.std()
        b_std = BCD.std()
        a_max = A.max()
        b_max = BCD.max()

        A_freq = A.mean(axis=1)  # (3,256)
        B_freq = BCD.mean(axis=1)
        freq_contrast = A_freq.mean(axis=0) - B_freq.mean(axis=0)
        fc_mean = freq_contrast.mean()
        fc_std = freq_contrast.std()
        fc_max = freq_contrast.max()
        fc_min = freq_contrast.min()

        return np.concatenate(
            [
                means,
                stds,
                maxs,
                np.array(
                    [
                        a_mean,
                        b_mean,
                        a_std,
                        b_std,
                        a_max,
                        b_max,
                        a_mean - b_mean,
                        a_max - b_max,
                    ],
                    dtype=np.float32,
                ),
                np.array([fc_mean, fc_std, fc_max, fc_min], dtype=np.float32),
            ],
            axis=0,
        )

    def _featurize_worker(task):
        i, p = task
        return i, extract_features_from_path(p)

    def featurize_paths_in_order(paths, max_workers=None, chunksize=64):
        n = len(paths)
        if n == 0:
            return np.empty((0, 0), dtype=np.float32)

        d0 = extract_features_from_path(paths[0]).shape[0]
        X_out = np.empty((n, d0), dtype=np.float32)

        if max_workers is None:
            cpu = os.cpu_count() or 2
            max_workers = min(8, max(2, cpu))

        tasks = list(enumerate(paths))

        try:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=max_workers, maxtasksperchild=512) as pool:
                for i, feat in pool.imap_unordered(
                    _featurize_worker, tasks, chunksize=chunksize
                ):
                    X_out[i] = feat
        except Exception as e:
            print(
                f"[WARN] Multiprocessing featurization failed ({type(e).__name__}: {e}). Falling back to sequential."
            )
            for i, p in tasks:
                X_out[i] = extract_features_from_path(p)

        return X_out

    train_ids = labels["id"].to_numpy()
    train_paths = paths_for_ids(train_dir, train_ids)
    X = featurize_paths_in_order(train_paths)
    y = labels["target"].astype(int).to_numpy()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y), dtype=np.float64)

    models = []
    for tr_idx, va_idx in skf.split(X, y):
        model = LogisticRegression(
            solver="lbfgs",
            max_iter=500,
            n_jobs=None,
            class_weight=None,
        )
        model.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = model.predict_proba(X[va_idx])[:, 1]
        models.append(model)

    test_ids = sample_sub["id"].to_numpy()
    test_paths = paths_for_ids(test_dir, test_ids)

    missing = [tid for tid, p in zip(test_ids, test_paths) if not os.path.exists(p)]
    if len(missing) > 0:
        raise FileNotFoundError(
            f"Missing {len(missing)} test .npy files (e.g., {missing[:3]})."
        )

    X_test = featurize_paths_in_order(test_paths)
    pred = np.mean([m.predict_proba(X_test)[:, 1] for m in models], axis=0)

    submission = pd.DataFrame({"id": test_ids, "target": np.clip(pred, 0.0, 1.0)})



## === cell 4
if submission is None:
    raise RuntimeError("submission was not created; cannot write submission.csv")

submission = submission[["id", "target"]].copy()

sample_sub_candidates = [
    "/kaggle/input/seti-breakthrough-listen/sample_submission.csv",
    "/kaggle/data/seti-breakthrough-listen/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
]
sample_path = next((p for p in sample_sub_candidates if os.path.exists(p)), None)
if sample_path is not None:
    ss = pd.read_csv(sample_path, usecols=["id"])
    submission = ss.merge(submission, on="id", how="left")
    if submission["target"].isna().any():
        submission["target"] = submission["target"].fillna(0.5)
    submission["target"] = submission["target"].astype(float).clip(0.0, 1.0)

submission.to_csv("submission.csv", index=False)

print(submission.head())
print(
    f"Wrote submission.csv with shape={submission.shape} and columns={submission.columns.tolist()}"
)
