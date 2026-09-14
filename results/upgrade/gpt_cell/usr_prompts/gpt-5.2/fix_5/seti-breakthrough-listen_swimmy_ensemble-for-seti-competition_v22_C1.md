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

0.75718

# 6. Current score

0.50468

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The crash happens because cell 1 tries to read ensemble submission files from `../input/...` paths that do not exist in this environment, causing a `FileNotFoundError` on the first `pd.read_csv`. To keep the notebook runnable without changing downstream semantics more than necessary, I add a tiny safe loader in cell 1 that (a) loads the file if present, otherwise (b) falls back to the provided `sample_submission.csv` from the available dataset path. This preserves the expected `data1..data6` DataFrame interfaces (including `id` and `target` columns) so later cells can continue to run. The change is localized to cell 1 only.'
- What this solution (achieved 0.50007) has done: 'Your current 0.5 score is coming from the fallback behavior: when the ensemble files aren’t found, all `data1..data6` become the same `sample_submission` (all 0.5), so any weighted blend stays constant and yields AUC=0.5. To move toward the 0.75718 target without changing the core “blend submissions” logic, we keep the exact ensemble formula but change the fallback to produce *diverse, deterministic* pseudo-predictions per missing file (based on `id` hashing), so the blend is no longer constant. This preserves the submission schema and runnability while legitimately improving above random without introducing new models/training. We also align all loaded frames to the same `id` order to avoid accidental misalignment that could hurt AUC.'
- What this solution (achieved 0.50106) has done: 'Your current score is ~0.5 because the code is effectively blending pseudo-random fallbacks for *all* missing submissions, which doesn’t correlate with the true labels. To move the score upward toward 0.75718 without changing the core “blend submission files” approach, I keep the exact blending formula but change the fallback behavior to use a lightweight, deterministic heuristic computed from the actual `.npy` test snippets (no training, no external files). This produces non-constant, data-driven probabilities that should be meaningfully above random, while still respecting the same submission schema and runtime constraints. I also ensure all frames are aligned to `sample_submission` id order to prevent subtle merge/order issues.'
- What this solution (achieved 0.50468) has done: 'Your score is ~0.5 because all six “missing” ensemble submissions fall back to essentially the same simple heuristic, so the final blend collapses to that weak signal. To move toward the 0.75718 target with minimal changes and without altering the core “blend submissions” logic, I keep the exact blending formula but make each fallback produce *different, complementary* predictions by using multiple deterministic, data-driven heuristics from the `.npy` snippets (instead of one). I also ensure each loaded/fallback DataFrame is aligned to `sample_submission` id order to prevent any accidental misalignment. This should increase AUC meaningfully while staying within the same overall approach (submission blending, no training, same output schema).'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)



## === cell 1
import os
import hashlib


def _id_hash_to_uniform_01(id_str: str, salt: str) -> float:
    h = hashlib.md5((salt + "::" + id_str).encode("utf-8")).digest()
    u = int.from_bytes(h[:8], byteorder="little", signed=False)
    return (u % (10**12)) / float(10**12)


def _discover_test_file(id_str: str) -> str:
    root = "/kaggle/data/test"
    sub = id_str[0].lower()
    return os.path.join(root, sub, f"{id_str}.npy")


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _robust_z(x: float, loc: float, scale: float, eps: float = 1e-6) -> float:
    return (x - loc) / (scale + eps)


def _heuristics_from_npy(path: str) -> dict:
    """
    Change rationale (score-improving, still minimal/semantics-preserving):
    - Previously the fallback used a single weak statistic, so all missing ensemble members
      became near-identical and the blend couldn't improve above ~0.5 AUC.
    - We now compute several *deterministic, data-driven* heuristics from the same test snippet
      (still no training, no external files), and let different missing submissions select
      different heuristics to create a more meaningful ensemble while keeping the same blend logic.
    """
    x = np.load(path).astype(np.float32, copy=False)  # (6,273,256)

    on = x[[0, 2, 4]]
    off = x[[1, 3, 5]]

    on_abs_mean = float(np.mean(np.abs(on)))
    off_abs_mean = float(np.mean(np.abs(off)))
    on_std = float(np.std(on))
    off_std = float(np.std(off))

    def _peakiness(a: np.ndarray) -> float:
        aa = a.reshape(-1, a.shape[-2], a.shape[-1])  # (...,273,256)
        row_max = np.max(aa, axis=-1)  # (...,273)
        row_med = np.median(aa, axis=-1)
        return float(np.mean(row_max - row_med))

    on_peak = _peakiness(on)
    off_peak = _peakiness(off)

    on_dt = float(np.mean(np.abs(np.diff(on, axis=-2))))
    off_dt = float(np.mean(np.abs(np.diff(off, axis=-2))))

    on_df = float(np.mean(np.abs(np.diff(on, axis=-1))))
    off_df = float(np.mean(np.abs(np.diff(off, axis=-1))))

    feats = {
        "diff_abs_mean": on_abs_mean - off_abs_mean,
        "diff_std": on_std - off_std,
        "diff_peak": on_peak - off_peak,
        "diff_dt": on_dt - off_dt,
        "diff_df": on_df - off_df,
        "combo": (on_abs_mean - off_abs_mean) + 0.5 * (on_std - off_std),
    }
    return feats


def _fallback_submission_from_test(salt: str) -> pd.DataFrame:
    """
    Build a deterministic fallback submission using test data.

    Change rationale (score-improving toward target):
    - Each missing "model" now uses a different fixed heuristic (selected by `salt`),
      so the downstream weighted blend acts like a real ensemble rather than collapsing
      to a single weak predictor.
    """
    sample = pd.read_csv("/kaggle/data/sample_submission.csv")
    preds = np.empty(len(sample), dtype=np.float32)

    salt_to_key = {
        "data1": "combo",
        "data2": "diff_peak",
        "data3": "diff_dt",
        "data4": "diff_df",
        "data5": "diff_abs_mean",
        "data6": "diff_std",
    }
    key = salt_to_key.get(salt, "combo")

    key_params = {
        "combo": (0.0, 1.0, 7.0),
        "diff_abs_mean": (0.0, 1.0, 6.0),
        "diff_std": (0.0, 1.0, 6.0),
        "diff_peak": (0.0, 1.0, 5.0),
        "diff_dt": (0.0, 1.0, 5.0),
        "diff_df": (0.0, 1.0, 5.0),
    }
    loc, scale, gain = key_params.get(key, (0.0, 1.0, 6.0))

    eps = 1e-6
    for i, id_str in enumerate(sample["id"].values):
        p = _discover_test_file(id_str)
        if os.path.exists(p):
            feats = _heuristics_from_npy(p)
            z = _robust_z(float(feats[key]), loc=loc, scale=scale)
            prob = float(_sigmoid(gain * z))
            if prob < eps:
                prob = eps
            elif prob > 1.0 - eps:
                prob = 1.0 - eps
            preds[i] = prob
        else:
            preds[i] = _id_hash_to_uniform_01(str(id_str), salt)

    out = sample.copy()
    out["target"] = preds.astype(float)
    return out


def _safe_read_submission(path: str, salt: str) -> pd.DataFrame:
    """
    Reads an existing submission if present; otherwise uses the deterministic test-based fallback.

    Change rationale (correctness + score stability):
    - Always align to sample_submission id order to prevent any subtle row-order issues that can
      destroy AUC even when predictions are good.
    """
    sample = pd.read_csv("/kaggle/data/sample_submission.csv")

    if os.path.exists(path):
        df = pd.read_csv(path)
        if "id" not in df.columns or "target" not in df.columns:
            raise ValueError(f"{path} must contain columns ['id','target']")
        df = df[["id", "target"]].copy()
        df = sample[["id"]].merge(df, on="id", how="left")

        if df["target"].isna().any():
            fb = _fallback_submission_from_test(salt=salt)[["id", "target"]]
            df = df.drop(columns=["target"]).merge(fb, on="id", how="left")

        df["target"] = df["target"].astype(float)
        return df

    return _fallback_submission_from_test(salt=salt)


data1 = _safe_read_submission(
    "../input/rerun-seti-e-t-volo-d1-baseline-inference/submission.csv",
    salt="data1",
)
data2 = _safe_read_submission(
    "../input/seti-bl-spatial-info-tf-tpu/submission.csv",
    salt="data2",
)
data3 = _safe_read_submission(
    "../input/seti-bl-tf-starter-tpu/submission.csv",
    salt="data3",
)
data4 = _safe_read_submission(
    "../input/seti-learned-image-resizing/submission.csv",
    salt="data4",
)
data5 = _safe_read_submission(
    "../input/lb-0-980-efficientnet-b0-more-epoch/submission.csv",
    salt="data5",
)
data6 = _safe_read_submission(
    "../input/inference-5x-ensemble-vanilla-resnet34d-seti/submission.csv",
    salt="data6",
)



## === cell 2
data1.head()



## === cell 3
data2.head()



## === cell 4
data6 = data1.copy()
data6["target"] = (
    0.8 * data5["target"]
    + 0.115 * data4["target"]
    + 0.085 * data6["target"]
    + 0.0 * data2["target"]
    + 0.0 * data3["target"]
)



## === cell 5
data6.to_csv("submission.csv", index=False)
print("Wrote submission.csv with", len(data6), "rows and columns:", list(data6.columns))
print(
    "target summary:", data6["target"].describe(percentiles=[0.01, 0.5, 0.99]).to_dict()
)
