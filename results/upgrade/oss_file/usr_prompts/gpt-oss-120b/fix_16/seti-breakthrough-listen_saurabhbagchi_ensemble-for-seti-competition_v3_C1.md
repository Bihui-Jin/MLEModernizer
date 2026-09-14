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

0.7275326111973154

# 6. Current score

0.49923

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The fix replaces the missing external submissions with the provided sample submission file, computes a simple baseline prediction (the overall positive rate from the training labels), fills the submission with this value, and writes a valid `submission.csv`. This resolves the file‑not‑found errors and ensures a correctly‑formatted output ready for Kaggle.'
- What this solution (achieved 0.49469) has done: 'Optimized the data loading by building a one‑time lookup table that maps each snippet id to its `.npy` path, eliminating the repeated full‑directory walk for every file. The lookup is cached for both train and test directories and reused in `compute_diff_feature`. This change preserves exact feature calculations and scaling while dramatically reducing I/O overhead, allowing the whole script to finish well within the 600 s limit.'
- What this solution (achieved 0.49741) has done: 'The fix moves the parallel feature extraction from a process pool (which cannot pickle the inner helper function) to a thread pool, eliminating the pickling error and allowing the script to run end‑to‑end and generate a valid `submission.csv`. No other logic is changed, preserving the original model and feature engineering.'
- What this solution (achieved 0.49923) has done: 'I keep the overall pipeline unchanged but add a few simple engineered features (the product of the mean difference and the ratio feature, and the square of the ratio feature) and train the logistic model a bit longer with the selected learning rate. These minor extensions often boost AUC without altering the core logic, helping move the score closer to the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import concurrent.futures

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"



## === cell 1
sample_paths = [
    "/kaggle/data/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
    "../input/sample_submission.csv",
    "./sample_submission.csv",
]
for p in sample_paths:
    if os.path.exists(p):
        sample_path = p
        break
else:
    raise FileNotFoundError("sample_submission.csv not found in expected locations")
submission_df = pd.read_csv(sample_path)



## === cell 2
train_label_paths = [
    "/kaggle/data/train_labels.csv",
    "/kaggle/input/train_labels.csv",
    "../input/train_labels.csv",
    "./train_labels.csv",
]
for p in train_label_paths:
    if os.path.exists(p):
        train_label_path = p
        break
else:
    raise FileNotFoundError("train_labels.csv not found in expected locations")
train_labels = pd.read_csv(train_label_path)




## === cell 3
def build_path_cache(root_dir: str) -> dict:
    cache = {}
    for subdir, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith(".npy"):
                cache[f[:-4]] = os.path.join(subdir, f)
    return cache


possible_data_dirs = [
    "/kaggle/data/train",
    "/kaggle/input/train",
    "../input/train",
    "./train",
]
for d in possible_data_dirs:
    if os.path.isdir(d):
        train_dir = d
        break
else:
    raise FileNotFoundError("train directory not found in expected locations")

_train_path_cache = build_path_cache(train_dir)


def compute_features(ids, cache):
    """
    Return a (n,4) array with:
      [mean_diff, var_diff, overall_mean, overall_var] for each id.
    """
    n = len(ids)

    def process_one(idx_id):
        idx, cur_id = idx_id
        arr = np.load(cache[cur_id], mmap_mode="r").astype(
            np.float32
        )  # shape (6,273,256)

        pos_means = arr.mean(axis=(1, 2), dtype=np.float32)
        a_mean = pos_means[[0, 2, 4]].mean()
        b_mean = pos_means[[1, 3, 5]].mean()
        diff = a_mean - b_mean

        pos_vars = arr.var(axis=(1, 2), dtype=np.float32)
        a_var = pos_vars[[0, 2, 4]].mean()
        b_var = pos_vars[[1, 3, 5]].mean()
        var_diff = a_var - b_var

        overall_mean = arr.mean(dtype=np.float32)
        overall_var = arr.var(dtype=np.float32)

        return idx, diff, var_diff, overall_mean, overall_var

    diffs = np.empty(n, dtype=np.float32)
    var_diffs = np.empty(n, dtype=np.float32)
    overall_means = np.empty(n, dtype=np.float32)
    overall_vars = np.empty(n, dtype=np.float32)

    indexed_ids = list(enumerate(ids))

    max_workers = os.cpu_count() or 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, diff, var_diff, o_mean, o_var in executor.map(
            process_one, indexed_ids, chunksize=64
        ):
            diffs[idx] = diff
            var_diffs[idx] = var_diff
            overall_means[idx] = o_mean
            overall_vars[idx] = o_var

    return np.vstack((diffs, var_diffs, overall_means, overall_vars)).T




## === cell 4
train_ids = train_labels["id"].values
train_targets = train_labels["target"].values.astype(np.float32)

train_feats = compute_features(train_ids, _train_path_cache)

ratio_feat = train_feats[:, 0] / (train_feats[:, 1] + 1e-6)
interaction = train_feats[:, 0] * train_feats[:, 1]  # diff * var_diff
squared_feats = train_feats**2  # element‑wise squares

ratio_times_diff = ratio_feat * train_feats[:, 0]  # ratio * diff
ratio_squared = ratio_feat**2  # ratio squared

train_feats = np.column_stack(
    (
        train_feats,
        ratio_feat[:, None],
        squared_feats,
        interaction[:, None],
        ratio_times_diff[:, None],
        ratio_squared[:, None],
    )
)

feat_mean = train_feats.mean(axis=0, keepdims=True)
feat_std = train_feats.std(axis=0, keepdims=True) + 1e-8
train_feats_std = (train_feats - feat_mean) / feat_std


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fit_logistic(X, y, sample_weight=None, lr=0.05, epochs=2000, l2_reg=1e-7):
    """Plain gradient‑descent logistic regression with optional sample weights."""
    if sample_weight is None:
        sample_weight = np.ones_like(y, dtype=np.float32)
    w = np.zeros(X.shape[1], dtype=np.float32)
    n = X.shape[0]
    for _ in range(epochs):
        z = X @ w
        p = sigmoid(z)
        grad = (X.T @ ((p - y) * sample_weight)) / n + l2_reg * w
        w -= lr * grad
    return w


def auc_score(y_true, y_score):
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    tp = np.cumsum(y_true_sorted)
    fp = np.cumsum(1 - y_true_sorted)
    tpr = tp / (tp[-1] + 1e-12)
    fpr = fp / (fp[-1] + 1e-12)
    return np.trapz(tpr, fpr)


rng = np.random.default_rng(42)
perm = rng.permutation(len(train_ids))
split = int(0.8 * len(train_ids))
train_idx, val_idx = perm[:split], perm[split:]

X_train = np.column_stack((np.ones(train_feats_std.shape[0]), train_feats_std))

pos_cnt = train_targets.sum()
neg_cnt = len(train_targets) - pos_cnt
weight_pos = len(train_targets) / (2 * pos_cnt + 1e-12)
weight_neg = len(train_targets) / (2 * neg_cnt + 1e-12)
sample_weights = np.where(train_targets == 1, weight_pos, weight_neg).astype(np.float32)

candidate_lrs = [0.01, 0.02, 0.05, 0.10, 0.20]
best_lr, best_auc = None, -1.0
for lr in candidate_lrs:
    w_tmp = fit_logistic(
        X_train[train_idx],
        train_targets[train_idx],
        sample_weight=sample_weights[train_idx],
        lr=lr,
        epochs=4000,  # more iterations for a stable estimate
        l2_reg=1e-7,
    )
    val_probs = sigmoid(X_train[val_idx] @ w_tmp)
    val_auc = auc_score(train_targets[val_idx], val_probs)
    if val_auc > best_auc:
        best_auc, best_lr = val_auc, lr

print(
    f"Validation AUC per LR: {[(lr, round(auc_score(train_targets[val_idx], sigmoid(X_train[val_idx] @ fit_logistic(X_train[train_idx], train_targets[train_idx], sample_weight=sample_weights[train_idx], lr=lr, epochs=4000, l2_reg=1e-7))),5)) for lr in candidate_lrs]}"
)
print(f"Chosen LR = {best_lr:.3f} with Validation AUC = {best_auc:.5f}")

w = fit_logistic(
    X_train,
    train_targets,
    sample_weight=sample_weights,
    lr=best_lr,
    epochs=8000,
    l2_reg=1e-7,
)



## === cell 5
possible_test_dirs = [
    "/kaggle/data/test",
    "/kaggle/input/test",
    "../input/test",
    "./test",
]
for d in possible_test_dirs:
    if os.path.isdir(d):
        test_dir = d
        break
else:
    raise FileNotFoundError("test directory not found in expected locations")

_test_path_cache = build_path_cache(test_dir)

test_ids = submission_df["id"].values
test_feats = compute_features(test_ids, _test_path_cache)

ratio_feat_test = test_feats[:, 0] / (test_feats[:, 1] + 1e-6)
interaction_test = test_feats[:, 0] * test_feats[:, 1]
squared_test = test_feats**2
ratio_times_diff_test = ratio_feat_test * test_feats[:, 0]
ratio_squared_test = ratio_feat_test**2

test_feats = np.column_stack(
    (
        test_feats,
        ratio_feat_test[:, None],
        squared_test,
        interaction_test[:, None],
        ratio_times_diff_test[:, None],
        ratio_squared_test[:, None],
    )
)

test_feats_std = (test_feats - feat_mean) / feat_std
X_test = np.column_stack((np.ones(test_feats_std.shape[0]), test_feats_std))
test_probs = sigmoid(X_test @ w)

submission_df["target"] = test_probs
submission_df.to_csv("submission.csv", index=False)
