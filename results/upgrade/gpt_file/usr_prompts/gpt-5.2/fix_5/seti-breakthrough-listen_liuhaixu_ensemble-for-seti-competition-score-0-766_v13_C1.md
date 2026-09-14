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

0.7625534441480373

# 6. Current score

0.4959

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I remove the hard dependency on external Kaggle dataset paths that don’t exist in your environment (the cause of the FileNotFoundError) and replace it with a robust ensembling loader that uses any available `sample_submission.csv` plus any local `submission*.csv` files if present. If none of the external submissions are available, the notebook still run end-to-end by producing a valid baseline submission (constant 0.5 probabilities) in the required `id,target` format. I also fix the incorrect weighted-sum logic (weights summed to >1 and ignored `data7`) to a safe normalized-average blend when multiple submissions exist, while preserving the “blend submissions” core intent. Finally, I add strict validation to ensure ID alignment and correct output shape before writing `submission.csv`.'
- What this solution (achieved 0.49519) has done: 'Your current 0.5 score comes from emitting constant predictions when no other `submission*.csv` files are found, so the smallest legitimate improvement is to generate non-constant predictions directly from the provided training data and test snippets. To preserve “blend submissions” core intent, I keep your ensembling loader, but add a fallback that trains a simple sklearn LogisticRegression on lightweight, deterministic summary features extracted from each `(6,273,256)` snippet and then predicts probabilities for the test ids. This keeps the approach minimal (no deep nets, no new packages) and should move AUC up toward your target without relying on external datasets. The script still always writes a valid `submission.csv` with `id,target` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.49294) has done: 'Your current score (~0.495) suggests the fallback model is learning almost nothing; the most common cause here is that raw per-snippet summary statistics are not on a comparable scale across features, so LogisticRegression underfits badly. I keep your exact feature extraction and LogisticRegression core logic, but add a minimal `StandardScaler` fit on train features and applied to test features to make optimization well-conditioned. I also set `C` slightly higher and `max_iter` a bit larger to better fit without changing the model family, and add a quick in-script train AUC check to confirm the model is learning (no metric leakage, just a sanity check). The submission format and ID alignment remain unchanged.'
- What this solution (achieved 0.4959) has done: 'Your current score indicates the fallback model still isn’t capturing the key “A vs off-target” cadence pattern, so we minimally strengthen the *same* LogisticRegression approach by making features more cadence-aware without changing the training loop or model family. Specifically, we keep your existing per-panel summary stats, but add a small set of additional handcrafted features that measure (1) consistency across the three A panels vs off panels and (2) robust “excess in A” signals (median/quantiles and variability), which are commonly predictive for this competition. We also add `class_weight="balanced"` to address class imbalance (common in SETI) while keeping the same solver and objective, which typically improves AUC when the model was near-random. Everything else (paths, ensembling behavior, output format, and always writing `submission.csv`) remains unchanged.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd



## === cell 1
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


def _read_submission_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"id", "target"}.issubset(df.columns):
        raise ValueError(
            f"{path} missing required columns; found {df.columns.tolist()}"
        )
    df = df[["id", "target"]].copy()
    df["id"] = df["id"].astype(str)
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    if df["target"].isna().any():
        raise ValueError(f"{path} has non-numeric target values.")
    return df


ref_paths = [
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
    "/kaggle/data/seti-breakthrough-listen/sample_submission.csv",
    "../input/sample_submission.csv",
    "../input/seti-breakthrough-listen/sample_submission.csv",
]
ref_path = next((p for p in ref_paths if os.path.exists(p)), None)
if ref_path is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv in expected locations."
    )

sample_sub = pd.read_csv(ref_path)[["id", "target"]].copy()
sample_sub["id"] = sample_sub["id"].astype(str)



## === cell 2
candidate_globs = [
    "/kaggle/working/submission*.csv",
    "/kaggle/working/*submission*.csv",
    "/kaggle/input/*/submission*.csv",
    "../input/*/submission*.csv",
]
candidate_paths = []
for pattern in candidate_globs:
    candidate_paths.extend(glob.glob(pattern))

candidate_paths = sorted(
    set(
        p
        for p in candidate_paths
        if os.path.basename(p) not in {"sample_submission.csv", "submission.csv"}
    )
)

loaded = []
for p in candidate_paths:
    try:
        df = _read_submission_csv(p)
        m = sample_sub[["id"]].merge(df, on="id", how="left")
        if m["target"].isna().any():
            continue
        loaded.append((p, m["target"].to_numpy(dtype=np.float64)))
    except Exception:
        continue

print(f"Found {len(loaded)} candidate submission(s) to blend.")
for p, _ in loaded[:10]:
    print(" -", p)




## === cell 3
def _find_first_existing(paths):
    return next((p for p in paths if os.path.exists(p)), None)


train_root = _find_first_existing(
    [
        "/kaggle/data/train",
        "/kaggle/data/seti-breakthrough-listen/train",
        "/kaggle/input/train",
        "/kaggle/input/seti-breakthrough-listen/train",
        "../input/train",
        "../input/seti-breakthrough-listen/train",
    ]
)
test_root = _find_first_existing(
    [
        "/kaggle/data/test",
        "/kaggle/data/seti-breakthrough-listen/test",
        "/kaggle/input/test",
        "/kaggle/input/seti-breakthrough-listen/test",
        "../input/test",
        "../input/seti-breakthrough-listen/test",
    ]
)
labels_path = _find_first_existing(
    [
        "/kaggle/data/train_labels.csv",
        "/kaggle/data/seti-breakthrough-listen/train_labels.csv",
        "/kaggle/input/train_labels.csv",
        "/kaggle/input/seti-breakthrough-listen/train_labels.csv",
        "../input/train_labels.csv",
        "../input/seti-breakthrough-listen/train_labels.csv",
    ]
)


def _build_id_to_npy_path(root_dir: str) -> dict:
    id2path = {}
    for fp in glob.glob(os.path.join(root_dir, "*", "*.npy")):
        fid = os.path.splitext(os.path.basename(fp))[0]
        id2path[fid] = fp
    return id2path


def _extract_features_from_array(arr: np.ndarray) -> np.ndarray:
    """
    arr: (6, 273, 256)

    Change rationale (score improvement toward target):
    - Keep the same "summary-statistics + A vs off-target differences" core logic,
      but add a minimal set of cadence-aware features that better reflect the ABACAD pattern:
        * consistency across A panels vs off panels (std across the 3 A panels / 3 off panels)
        * robust excess-in-A stats (median and quantiles differences)
      These are still lightweight deterministic summaries; no change to model family or training loop.
    """
    x = arr.astype(np.float32, copy=False)

    panel_mean = x.mean(axis=(1, 2))  # (6,)
    panel_std = x.std(axis=(1, 2))  # (6,)
    panel_max = x.max(axis=(1, 2))  # (6,)
    panel_min = x.min(axis=(1, 2))  # (6,)
    panel_absmean = np.abs(x).mean(axis=(1, 2))  # (6,)
    panel_energy = (x * x).mean(axis=(1, 2))  # (6,)

    A_idx = np.array([0, 2, 4])
    O_idx = np.array([1, 3, 5])

    A_mean = panel_mean[A_idx].mean()
    O_mean = panel_mean[O_idx].mean()
    A_energy = panel_energy[A_idx].mean()
    O_energy = panel_energy[O_idx].mean()
    A_absmean = panel_absmean[A_idx].mean()
    O_absmean = panel_absmean[O_idx].mean()

    A_mean_std = panel_mean[A_idx].std()
    O_mean_std = panel_mean[O_idx].std()
    A_energy_std = panel_energy[A_idx].std()
    O_energy_std = panel_energy[O_idx].std()
    A_absmean_std = panel_absmean[A_idx].std()
    O_absmean_std = panel_absmean[O_idx].std()

    A_mean_med = np.median(panel_mean[A_idx])
    O_mean_med = np.median(panel_mean[O_idx])
    A_energy_med = np.median(panel_energy[A_idx])
    O_energy_med = np.median(panel_energy[O_idx])

    A_mean_q75 = np.quantile(panel_mean[A_idx], 0.75)
    O_mean_q75 = np.quantile(panel_mean[O_idx], 0.75)
    A_energy_q75 = np.quantile(panel_energy[A_idx], 0.75)
    O_energy_q75 = np.quantile(panel_energy[O_idx], 0.75)

    A_panel_max_of_max = panel_max[A_idx].max()
    O_panel_max_of_max = panel_max[O_idx].max()

    base = np.concatenate(
        [
            panel_mean,
            panel_std,
            panel_max,
            panel_min,
            panel_absmean,
            panel_energy,
            np.array(
                [
                    A_mean,
                    O_mean,
                    A_mean - O_mean,
                    A_energy,
                    O_energy,
                    A_energy - O_energy,
                    A_absmean,
                    O_absmean,
                    A_absmean - O_absmean,
                ],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32, copy=False)

    extra = np.array(
        [
            A_mean_std,
            O_mean_std,
            A_mean_std - O_mean_std,
            A_energy_std,
            O_energy_std,
            A_energy_std - O_energy_std,
            A_absmean_std,
            O_absmean_std,
            A_absmean_std - O_absmean_std,
            A_mean_med,
            O_mean_med,
            A_mean_med - O_mean_med,
            A_energy_med,
            O_energy_med,
            A_energy_med - O_energy_med,
            A_mean_q75,
            O_mean_q75,
            A_mean_q75 - O_mean_q75,
            A_energy_q75,
            O_energy_q75,
            A_energy_q75 - O_energy_q75,
            A_panel_max_of_max,
            O_panel_max_of_max,
            A_panel_max_of_max - O_panel_max_of_max,
        ],
        dtype=np.float32,
    )

    return np.concatenate([base, extra]).astype(np.float32, copy=False)


def _make_feature_matrix(ids, id2path):
    n_feats = (6 * 6 + 9) + 24
    X = np.zeros((len(ids), n_feats), dtype=np.float32)
    for i, fid in enumerate(ids):
        fp = id2path.get(fid)
        if fp is None:
            raise FileNotFoundError(f"Missing .npy for id={fid}")
        arr = np.load(fp)  # float16 -> feature fn casts to float32
        X[i] = _extract_features_from_array(arr)
    return X


fallback_pred = None
if len(loaded) == 0:
    if train_root is None or test_root is None or labels_path is None:
        raise FileNotFoundError(
            "No external submissions found and could not locate train/test/labels to build a model fallback."
        )

    labels = pd.read_csv(labels_path)
    labels["id"] = labels["id"].astype(str)

    train_id2path = _build_id_to_npy_path(train_root)
    test_id2path = _build_id_to_npy_path(test_root)

    train_ids = labels["id"].tolist()
    train_ids = [fid for fid in train_ids if fid in train_id2path]
    y = labels.set_index("id").loc[train_ids, "target"].to_numpy(dtype=np.int32)

    test_ids = sample_sub["id"].tolist()
    missing_test = [fid for fid in test_ids if fid not in test_id2path]
    if missing_test:
        raise FileNotFoundError(
            f"Missing {len(missing_test)} test .npy files; first few: {missing_test[:5]}"
        )

    print("Building features:")
    print(
        " - train_root:",
        train_root,
        "train files:",
        len(train_id2path),
        "using:",
        len(train_ids),
    )
    print(
        " - test_root :",
        test_root,
        "test files :",
        len(test_id2path),
        "predicting:",
        len(test_ids),
    )

    X_train = _make_feature_matrix(train_ids, train_id2path)
    X_test = _make_feature_matrix(test_ids, test_id2path)

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=800,
        n_jobs=None,
        random_state=0,
        class_weight="balanced",
        C=3.0,
    )
    clf.fit(X_train_s, y)

    try:
        train_pred = clf.predict_proba(X_train_s)[:, 1]
        print("Train AUC (sanity check):", roc_auc_score(y, train_pred))
    except Exception as e:
        print("Could not compute train AUC sanity check:", repr(e))

    fallback_pred = clf.predict_proba(X_test_s)[:, 1].astype(np.float64)



## === cell 4
if len(loaded) > 0:
    preds = np.vstack([arr for _, arr in loaded])
    pred = preds.mean(axis=0)
else:
    pred = fallback_pred

pred = np.clip(pred, 0.0, 1.0)
data1 = pd.DataFrame({"id": sample_sub["id"].values, "target": pred})



## === cell 5
data11 = data1.copy()



## === cell 6
data11 = data11[["id", "target"]].copy()

if len(data11) != len(sample_sub):
    raise ValueError(
        f"Submission length mismatch: {len(data11)} vs expected {len(sample_sub)}"
    )
if data11["id"].duplicated().any():
    raise ValueError("Duplicate ids found in submission.")
if not np.isfinite(data11["target"]).all():
    raise ValueError("Non-finite target values found in submission.")



## === cell 7
data11.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", data11.shape)
print(data11.head())
print("target summary:", data11["target"].describe())
