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
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor

np.random.seed(42)

root_path = Path("../input")
sample_path = root_path / "sample_submission.csv"
train_labels_path = root_path / "train_labels.csv"
train_dir = root_path / "train"
test_dir = root_path / "test"

submission_df = pd.read_csv(sample_path)
train_labels = pd.read_csv(train_labels_path)


def build_id_path_map(base_dir: Path) -> dict:
    """Scan sub‑folders (0‑15) and map snippet id → file path."""
    id_path = {}
    for sub in map(str, range(16)):
        folder = base_dir / sub
        for p in folder.glob("*.npy"):
            id_path[p.stem] = p
    return id_path


train_id_path_map = build_id_path_map(train_dir)
test_id_path_map = build_id_path_map(test_dir)


def _compute_feature(sid_path):
    """Return an enriched statistical feature vector for one snippet."""
    sid, path = sid_path
    if path is None:
        return np.full(50, np.nan, dtype=np.float32)

    arr = np.load(path)  # (6, 273, 256), float16

    overall_mean = float(arr.mean())
    overall_max = float(arr.max())
    overall_min = float(arr.min())
    overall_std = float(arr.std())
    overall_median = float(np.median(arr))
    overall_range = overall_max - overall_min
    overall_skew = overall_mean - overall_median
    overall_cv = overall_std / (overall_mean + 1e-6)  # new feature

    slice_means = arr.mean(axis=(1, 2)).astype(float)  # (6,)
    slice_maxs = arr.max(axis=(1, 2)).astype(float)
    slice_mins = arr.min(axis=(1, 2)).astype(float)
    slice_stds = arr.std(axis=(1, 2)).astype(float)
    slice_medians = np.median(arr, axis=(1, 2)).astype(float)

    a_idx = [0, 2, 4]
    non_a_idx = [1, 3, 5]

    diff_a_non_a_mean = float(slice_means[a_idx].mean() - slice_means[non_a_idx].mean())
    diff_a_non_a_max = float(slice_maxs[a_idx].mean() - slice_maxs[non_a_idx].mean())
    diff_a_non_a_std = float(slice_stds[a_idx].mean() - slice_stds[non_a_idx].mean())
    diff_a_non_a_median = float(
        slice_medians[a_idx].mean() - slice_medians[non_a_idx].mean()
    )
    diff_a_non_a_min = float(slice_mins[a_idx].mean() - slice_mins[non_a_idx].mean())
    slice_ranges = slice_maxs - slice_mins
    diff_a_non_a_range = float(
        slice_ranges[a_idx].mean() - slice_ranges[non_a_idx].mean()
    )

    prop_above_mean = float(np.mean(arr > overall_mean))
    prop_above_median = float(np.mean(arr > overall_median))
    q90 = float(np.quantile(arr, 0.90))
    prop_above_90 = float(np.mean(arr > q90))
    q95 = float(np.quantile(arr, 0.95))
    prop_above_95 = float(np.mean(arr > q95))

    slice_prop_above_mean = float(np.mean(arr > slice_means[:, None, None]))
    slice_prop_above_90 = float(
        np.mean(arr > np.quantile(arr, 0.90, axis=(1, 2))[:, None, None])
    )

    feat = np.concatenate(
        [
            [
                overall_mean,
                overall_max,
                overall_min,
                overall_std,
                overall_median,
                overall_range,
                overall_skew,
                overall_cv,  # new
            ],
            slice_means,
            slice_maxs,
            slice_mins,
            slice_stds,
            slice_medians,
            [
                diff_a_non_a_mean,
                diff_a_non_a_max,
                diff_a_non_a_std,
                diff_a_non_a_median,
                diff_a_non_a_min,
                diff_a_non_a_range,
                prop_above_mean,
                prop_above_median,
                prop_above_90,
                prop_above_95,
                slice_prop_above_mean,  # new
                slice_prop_above_90,  # new
            ],
        ]
    ).astype(np.float32)

    feat[~np.isfinite(feat)] = np.nan
    return feat


def extract_features(ids, id_path_map):
    """Compute feature matrix (n_samples, 50) for given ids."""
    lookup = [(sid, id_path_map.get(sid)) for sid in ids]
    with ThreadPoolExecutor(max_workers=5) as executor:
        feats = list(executor.map(_compute_feature, lookup))
    return np.vstack(feats)


train_ids = train_labels["id"].tolist()
X_train = extract_features(train_ids, train_id_path_map)
y_train = train_labels["target"].values

imputer = SimpleImputer(strategy="median")
X_train_imputed = imputer.fit_transform(X_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
)

best_auc = 0.0
best_params = {"C": 1.0, "solver": "lbfgs"}

c_candidates = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
solvers = ["lbfgs", "liblinear"]

for C in c_candidates:
    for sol in solvers:
        clf = LogisticRegression(
            max_iter=2000,
            n_jobs=5,
            class_weight="balanced",
            C=C,
            solver=sol,
        )
        clf.fit(X_tr, y_tr)
        val_auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
        if val_auc > best_auc:
            best_auc = val_auc
            best_params = {"C": C, "solver": sol}

print(
    f"Best validation AUC={best_auc:.5f} achieved with C={best_params['C']}, solver={best_params['solver']}"
)

model = LogisticRegression(
    max_iter=2000,
    n_jobs=5,
    class_weight="balanced",
    C=best_params["C"],
    solver=best_params["solver"],
)
model.fit(X_train_scaled, y_train)

train_auc = roc_auc_score(y_train, model.predict_proba(X_train_scaled)[:, 1])
print(f"Training AUC (full data): {train_auc:.5f}")



## === cell 1
test_ids = submission_df["id"].tolist()
X_test = extract_features(test_ids, test_id_path_map)

X_test_imputed = imputer.transform(X_test)
X_test_scaled = scaler.transform(X_test_imputed)

test_probs = model.predict_proba(X_test_scaled)[:, 1]
submission_df["target"] = test_probs



## === cell 2
output_path = Path("submission.csv")
submission_df.to_csv(output_path, index=False)
print(
    f"Submission written to {output_path.resolve()}. "
    f"Mean predicted probability: {submission_df['target'].mean():.5f}"
)
