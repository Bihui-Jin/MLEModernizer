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

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _find_submission_like_csvs(
    search_roots, max_files=50, max_glob_hits_per_pattern=2000
):
    found = []
    for root in search_roots:
        if not os.path.exists(root):
            continue
        patterns = [
            os.path.join(root, "**", "submission.csv"),
            os.path.join(root, "**", "*submission*.csv"),
        ]
        for pat in patterns:
            hits = glob.iglob(pat, recursive=True)
            for j, p in enumerate(hits):
                if j >= max_glob_hits_per_pattern:
                    break
                if os.path.basename(p).lower() == "sample_submission.csv":
                    continue
                found.append(p)

    dedup = []
    seen = set()
    for p in found:
        rp = os.path.realpath(p)
        if rp not in seen:
            seen.add(rp)
            dedup.append(p)
        if len(dedup) >= max_files:
            break
    return dedup[:max_files]


def _load_pred_csv(path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "id" not in cols or "target" not in cols:
        return None
    df = df.rename(columns={cols["id"]: "id", cols["target"]: "target"})
    df = df[["id", "target"]].copy()
    df["id"] = df["id"].astype(str)
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df = df.dropna(subset=["target"])
    return df


search_roots = [
    "/kaggle/working",
    "../working",
]

candidate_paths = _find_submission_like_csvs(search_roots, max_files=100)

pred_dfs = []
pred_paths_used = []
for p in candidate_paths:
    df = _load_pred_csv(p)
    if df is None:
        continue
    if len(df) < 1000:
        continue
    pred_dfs.append(df)
    pred_paths_used.append(p)

print(f"Found {len(pred_dfs)} usable submission-like CSV(s).")
for i, p in enumerate(pred_paths_used[:20], 1):
    print(f"{i:02d}: {p}")
if len(pred_paths_used) > 20:
    print(f"... ({len(pred_paths_used)-20} more)")




## === cell 1
sample_paths = [
    "/kaggle/data/sample_submission.csv",
    "/kaggle/data/seti-breakthrough-listen/sample_submission.csv",
    "/kaggle/input/seti-breakthrough-listen/sample_submission.csv",
    "../input/seti-breakthrough-listen/sample_submission.csv",
    "../input/sample_submission.csv",
]
sample_path = None
for p in sample_paths:
    if os.path.exists(p):
        sample_path = p
        break

if sample_path is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv in expected Kaggle paths."
    )

sample_sub = pd.read_csv(sample_path)
sample_sub["id"] = sample_sub["id"].astype(str)
data11 = sample_sub[["id"]].copy()




## === cell 2
order = np.argsort(np.array(pred_paths_used, dtype=str))
pred_dfs = [pred_dfs[i] for i in order]
pred_paths_used = [pred_paths_used[i] for i in order]

aligned_preds = []
for df in pred_dfs:
    tmp = data11.merge(df, on="id", how="left", validate="one_to_one")
    aligned_preds.append(tmp["target"].to_numpy(dtype=np.float64))

n_models = len(aligned_preds)

blended_from_found_files = None
if n_models > 0:
    P = np.vstack(aligned_preds)  # (n_models, n_samples)
    for i in range(n_models):
        col = P[i]
        if np.isnan(col).any():
            m = np.nanmean(col)
            if not np.isfinite(m):
                m = 0.5
            col = np.where(np.isnan(col), m, col)
            P[i] = col

    if n_models >= 6:
        w = np.array([0.11, 0.11, 0.11, 0.11, 0.11, 0.53], dtype=np.float64)
        w = w / w.sum()
        blended_from_found_files = (w[:, None] * P[:6]).sum(axis=0)
    else:
        blended_from_found_files = P.mean(axis=0)




## === cell 3
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from concurrent.futures import ThreadPoolExecutor


def _resolve_existing_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


train_labels_path = _resolve_existing_path(
    [
        "/kaggle/data/train_labels.csv",
        "/kaggle/data/seti-breakthrough-listen/train_labels.csv",
        "/kaggle/input/seti-breakthrough-listen/train_labels.csv",
        "../input/seti-breakthrough-listen/train_labels.csv",
        "../input/train_labels.csv",
    ]
)

train_root = _resolve_existing_path(
    [
        "/kaggle/data/train",
        "/kaggle/data/seti-breakthrough-listen/train",
        "/kaggle/input/seti-breakthrough-listen/train",
        "../input/seti-breakthrough-listen/train",
        "../input/train",
        "/kaggle/data/old_leaky_data/train_old",
        "/kaggle/data/seti-breakthrough-listen/old_leaky_data/train_old",
        "/kaggle/input/seti-breakthrough-listen/old_leaky_data/train_old",
        "../input/seti-breakthrough-listen/old_leaky_data/train_old",
        "../input/old_leaky_data/train_old",
    ]
)

test_root = _resolve_existing_path(
    [
        "/kaggle/data/test",
        "/kaggle/data/seti-breakthrough-listen/test",
        "/kaggle/input/seti-breakthrough-listen/test",
        "../input/seti-breakthrough-listen/test",
        "../input/test",
        "/kaggle/data/old_leaky_data/test_old",
        "/kaggle/data/seti-breakthrough-listen/old_leaky_data/test_old",
        "/kaggle/input/seti-breakthrough-listen/old_leaky_data/test_old",
        "../input/seti-breakthrough-listen/old_leaky_data/test_old",
        "../input/old_leaky_data/test_old",
    ]
)


def _extract_features(arr):
    x = arr.astype(np.float32, copy=False)

    A_panels = x[[0, 2, 4]]  # (3,273,256)
    B_panels = x[[1, 3, 5]]  # (3,273,256)

    A = A_panels.mean(axis=0)  # on-target average
    B = B_panels.mean(axis=0)  # off-target average

    D = A - B
    AD = np.abs(D)

    q90, q95, q99 = 0.90, 0.95, 0.99
    A_q = np.quantile(A, [q90, q99])
    B_q = np.quantile(B, [q90, q99])
    D_q = np.quantile(D, [q90, q99])
    AD_q = np.quantile(AD, [q90, q99])

    feats = [
        float(A.mean()),
        float(A.std()),
        float(A_q[0]),
        float(A_q[1]),
        float(B.mean()),
        float(B.std()),
        float(B_q[0]),
        float(B_q[1]),
        float(D.mean()),
        float(D.std()),
        float(D_q[0]),
        float(D_q[1]),
        float(AD.mean()),
        float(AD.std()),
        float(AD_q[0]),
        float(AD_q[1]),
    ]

    a_t = A.max(axis=1)  # (273,)
    a_f = A.max(axis=0)  # (256,)
    d_t = AD.max(axis=1)
    d_f = AD.max(axis=0)

    a_t_p95 = float(np.quantile(a_t, q95))
    a_f_p95 = float(np.quantile(a_f, q95))
    d_t_p95 = float(np.quantile(d_t, q95))
    d_f_p95 = float(np.quantile(d_f, q95))

    feats += [
        float(a_t.mean()),
        float(a_t.std()),
        a_t_p95,
        float(a_t.max()),
        float(a_f.mean()),
        float(a_f.std()),
        a_f_p95,
        float(a_f.max()),
        float(d_t.mean()),
        float(d_t.std()),
        d_t_p95,
        float(d_t.max()),
        float(d_f.mean()),
        float(d_f.std()),
        d_f_p95,
        float(d_f.max()),
    ]

    eps = 1e-6
    A2 = float(np.mean(A * A))
    B2 = float(np.mean(B * B))
    mean_AD = float(np.mean(AD))
    mean_absB = float(np.mean(np.abs(B)))
    feats += [
        float((A2 + eps) / (B2 + eps)),
        float((mean_AD + eps) / (mean_absB + eps)),
    ]

    diff_t = np.diff(A, axis=0)
    diff_f = np.diff(A, axis=1)
    diff_t_off = np.diff(B, axis=0)
    diff_f_off = np.diff(B, axis=1)

    g_on = float(np.mean(np.abs(diff_t)) + np.mean(np.abs(diff_f)))
    g_off = float(np.mean(np.abs(diff_t_off)) + np.mean(np.abs(diff_f_off)))
    feats += [
        g_on,
        g_off,
        float(g_on - g_off),
        float((g_on + eps) / (g_off + eps)),
    ]

    diag1 = float(np.abs(A[1:, 1:] - A[:-1, :-1]).mean())
    diag2 = float(np.abs(A[1:, :-1] - A[:-1, 1:]).mean())
    diag1_off = float(np.abs(B[1:, 1:] - B[:-1, :-1]).mean())
    diag2_off = float(np.abs(B[1:, :-1] - B[:-1, 1:]).mean())
    d_on = diag1 + diag2
    d_off = diag1_off + diag2_off
    feats += [
        float(d_on),
        float(d_off),
        float(d_on - d_off),
        float((d_on + eps) / (d_off + eps)),
    ]

    b_t = B.max(axis=1)
    b_f = B.max(axis=0)

    a_t_mean = float(a_t.mean())
    a_f_mean = float(a_f.mean())
    b_t_mean = float(b_t.mean())
    b_f_mean = float(b_f.mean())

    a_t_ratio = float(a_t.max() / (a_t_mean + eps))
    a_f_ratio = float(a_f.max() / (a_f_mean + eps))
    b_t_ratio = float(b_t.max() / (b_t_mean + eps))
    b_f_ratio = float(b_f.max() / (b_f_mean + eps))

    feats += [
        a_t_ratio,
        a_f_ratio,
        b_t_ratio,
        b_f_ratio,
        float(a_t_ratio - b_t_ratio),
        float(a_f_ratio - b_f_ratio),
    ]

    k10_t = 10 if d_t.size >= 10 else d_t.size
    k10_f = 10 if d_f.size >= 10 else d_f.size
    dt_top = np.partition(d_t, -k10_t)[-k10_t:]
    df_top = np.partition(d_f, -k10_f)[-k10_f:]
    feats += [
        float(dt_top.mean()),
        float(dt_top.max()),
        float(df_top.mean()),
        float(df_top.max()),
    ]

    flat = AD.reshape(-1)
    k = max(1, int(0.01 * flat.size))
    topk = np.partition(flat, -k)[-k:]
    feats += [
        float((float(topk.sum()) + eps) / (float(flat.sum()) + eps)),
    ]

    A_p_mean = A_panels.mean(axis=(1, 2))
    B_p_mean = B_panels.mean(axis=(1, 2))
    A_p_std = A_panels.std(axis=(1, 2))
    B_p_std = B_panels.std(axis=(1, 2))
    A_p_max = A_panels.max(axis=(1, 2))
    B_p_max = B_panels.max(axis=(1, 2))

    feats += [
        float(A_p_mean.mean()),
        float(A_p_mean.std()),
        float(B_p_mean.mean()),
        float(B_p_mean.std()),
        float(A_p_std.mean()),
        float(A_p_std.std()),
        float(B_p_std.mean()),
        float(B_p_std.std()),
        float(A_p_max.mean()),
        float(A_p_max.std()),
        float(B_p_max.mean()),
        float(B_p_max.std()),
        float((float(A_p_max.mean()) + eps) / (float(B_p_max.mean()) + eps)),
        float((float(A_p_mean.std()) + eps) / (float(B_p_mean.std()) + eps)),
    ]

    A_centered = A_panels - A_panels.mean(axis=(1, 2), keepdims=True)
    B_centered = B_panels - B_panels.mean(axis=(1, 2), keepdims=True)
    A_dot01 = float(np.mean(A_centered[0] * A_centered[1]))
    A_dot02 = float(np.mean(A_centered[0] * A_centered[2]))
    A_dot12 = float(np.mean(A_centered[1] * A_centered[2]))
    B_dot01 = float(np.mean(B_centered[0] * B_centered[1]))
    B_dot02 = float(np.mean(B_centered[0] * B_centered[2]))
    B_dot12 = float(np.mean(B_centered[1] * B_centered[2]))
    feats += [
        float((A_dot01 + A_dot02 + A_dot12) / 3.0),
        float((B_dot01 + B_dot02 + B_dot12) / 3.0),
        float(((A_dot01 + A_dot02 + A_dot12) - (B_dot01 + B_dot02 + B_dot12)) / 3.0),
    ]

    def _top_frac_panels(P):  # P: (3,273,256)
        P2 = P.reshape(P.shape[0], -1)
        kk = max(1, int(0.01 * P2.shape[1]))
        part = np.partition(P2, P2.shape[1] - kk, axis=1)[:, -kk:]
        num = part.sum(axis=1) + eps
        den = P2.sum(axis=1) + eps
        return (num / den).astype(np.float32, copy=False)

    A_top = _top_frac_panels(A_panels)
    B_top = _top_frac_panels(B_panels)
    feats += [
        float(A_top.mean()),
        float(B_top.mean()),
        float(A_top.mean() - B_top.mean()),
        float((float(A_top.mean()) + eps) / (float(B_top.mean()) + eps)),
    ]

    A_flat = A.reshape(-1)
    B_flat = B.reshape(-1)
    D_flat = D.reshape(-1)

    A_energy = float(np.mean(A_flat * A_flat))
    B_energy = float(np.mean(B_flat * B_flat))
    D_energy = float(np.mean(D_flat * D_flat))

    feats += [
        float(D_energy),
        float((D_energy + eps) / (A_energy + eps)),
        float((D_energy + eps) / (B_energy + eps)),
    ]

    def _mean_pairwise_corr(P):  # P: (3,273,256)
        P2 = P.reshape(3, -1).astype(np.float32, copy=False)
        P2 = P2 - P2.mean(axis=1, keepdims=True)
        denom = np.sqrt((P2 * P2).sum(axis=1, keepdims=False)) + eps
        Pn = P2 / denom[:, None]
        c01 = float((Pn[0] * Pn[1]).mean())
        c02 = float((Pn[0] * Pn[2]).mean())
        c12 = float((Pn[1] * Pn[2]).mean())
        return (c01 + c02 + c12) / 3.0

    A_corr = float(_mean_pairwise_corr(A_panels))
    B_corr = float(_mean_pairwise_corr(B_panels))
    feats += [
        A_corr,
        B_corr,
        float(A_corr - B_corr),
    ]

    feats += [
        float(A_p_max.std()),
        float(B_p_max.std()),
        float(A_p_max.std() - B_p_max.std()),
    ]

    return np.asarray(feats, dtype=np.float32)


def _list_available_ids(root_dir):
    mp = {}
    if root_dir is None or (not os.path.isdir(root_dir)):
        return mp
    for p in glob.iglob(os.path.join(root_dir, "**", "*.npy"), recursive=True):
        base = os.path.basename(p)
        if not base.endswith(".npy"):
            continue
        k = base[:-4]
        if k not in mp:
            mp[k] = p
    return mp


def _build_feature_matrix(ids, id2path, n_features, max_workers=None):
    X = np.full((len(ids), n_features), np.nan, dtype=np.float32)

    def _one(i, id_str):
        p = id2path.get(id_str)
        if p is None:
            return i, None, True
        arr = np.load(p, mmap_mode="r")
        return i, _extract_features(arr), False

    if max_workers is None:
        max_workers = min(8, max(1, os.cpu_count() or 2))

    miss_flags = np.zeros(len(ids), dtype=np.int8)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, feats, is_miss in ex.map(
            lambda t: _one(*t), enumerate(ids), chunksize=512
        ):
            if is_miss or feats is None:
                miss_flags[i] = 1
            else:
                X[i, :] = feats
    return X, int(miss_flags.sum())


if blended_from_found_files is None:
    if train_labels_path is None or train_root is None or test_root is None:
        data11["target"] = 0.5
    else:
        labels = pd.read_csv(train_labels_path)
        labels["id"] = labels["id"].astype(str)
        labels["target"] = labels["target"].astype(int)

        all_train_ids = labels["id"].values
        all_y = labels["target"].values.astype(int)
        test_ids = data11["id"].values

        train_id2path = _list_available_ids(train_root)
        test_id2path = _list_available_ids(test_root)

        train_keys = np.fromiter(train_id2path.keys(), dtype=object)
        train_mask = np.isin(all_train_ids, train_keys)
        train_ids = all_train_ids[train_mask]
        y = all_y[train_mask]

        print(
            f"Train IDs: labels={len(all_train_ids)}, available_files={len(train_ids)}; train_root={train_root}"
        )
        print(
            f"Test IDs (submission template)={len(test_ids)}; available_test_files={len(test_id2path)}; test_root={test_root}"
        )

        if len(train_ids) < 1000:
            data11["target"] = 0.5
        else:
            any_path = train_id2path.get(train_ids[0])
            if any_path is None:
                raise FileNotFoundError(
                    "Could not locate any training .npy file to infer feature size."
                )
            n_features = int(
                _extract_features(np.load(any_path, mmap_mode="r")).shape[0]
            )

            X_train, miss_tr = _build_feature_matrix(
                train_ids, train_id2path, n_features=n_features
            )
            X_test, miss_te = _build_feature_matrix(
                test_ids, test_id2path, n_features=n_features
            )

            print(
                f"Feature matrix built: X_train={X_train.shape} (missing {miss_tr}), X_test={X_test.shape} (missing {miss_te})"
            )

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            oof = np.zeros(len(train_ids), dtype=np.float64)
            test_pred = np.zeros(len(test_ids), dtype=np.float64)

            for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y), 1):
                Xtr_raw = X_train[tr_idx]
                Xva_raw = X_train[va_idx]
                Xte_raw = X_test

                col_mean = np.nanmean(Xtr_raw, axis=0)
                col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0).astype(
                    np.float32
                )

                Xtr_imp = np.where(np.isfinite(Xtr_raw), Xtr_raw, col_mean)
                Xva_imp = np.where(np.isfinite(Xva_raw), Xva_raw, col_mean)
                Xte_imp = np.where(np.isfinite(Xte_raw), Xte_raw, col_mean)

                scaler = StandardScaler()
                Xtr = scaler.fit_transform(Xtr_imp)
                Xva = scaler.transform(Xva_imp)
                Xte = scaler.transform(Xte_imp)

                model = LogisticRegression(
                    solver="lbfgs",
                    C=3.0,
                    max_iter=1500,
                    n_jobs=1,
                    random_state=42,
                    class_weight="balanced",
                )
                model.fit(Xtr, y[tr_idx])
                oof[va_idx] = model.predict_proba(Xva)[:, 1]
                test_pred += model.predict_proba(Xte)[:, 1] / skf.get_n_splits()
                auc = roc_auc_score(y[va_idx], oof[va_idx])
                print(f"Fold {fold} AUC: {auc:.5f}")

            full_auc = roc_auc_score(y, oof)
            print(f"OOF AUC (sanity check): {full_auc:.5f}")

            data11["target"] = np.clip(test_pred, 0.0, 1.0)
else:
    data11["target"] = np.clip(blended_from_found_files, 0.0, 1.0)




## === cell 4
out_path = "submission.csv"
data11.to_csv(out_path, index=False)

assert os.path.exists(out_path), "submission.csv was not created."
assert list(data11.columns) == [
    "id",
    "target",
], f"Unexpected submission columns: {data11.columns.tolist()}"
assert len(data11) == len(sample_sub), "Row count mismatch vs sample_submission."
print(f"Wrote {out_path} with {len(data11)} rows.")
print(data11.head())
