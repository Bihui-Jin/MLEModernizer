# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Overview
Develop a model to tag audio data automatically using a diverse vocabulary of 80 categories.

## Metric
The task consists of predicting the audio labels (tags) for every test clip. Some test clips bear one label while others bear several labels. The predictions are to be done at the clip level, i.e., no start/end timestamps for the sound events are required.

The primary metric is label-weighted label-ranking average precision. 

The  "label-weighted" part means that the overall score is the average over all the *labels* in the test set, where each label receives equal weight (by contrast, plain *lrap* gives each *test item* equal weight).

## Submission Format
For each `fname` in the test set, you must predict the probability of each label. The file should contain a header and have the following format:

```
fname,Accelerating_and_revving_and_vroom,...Zipper_(clothing)
000ccb97.wav,0.1,....,0.3
0012633b.wav,0.0,...,0.8
```

## Dataset
The following 5 audio files in the curated train set have a wrong label, due to a bug in the file renaming process:\
`f76181c4.wav, 77b925c2.wav, 6a1f682a.wav, c7db12aa.wav, 7752cc8a.wav`

The audio file `1d44b0bd.wav` in the curated train set was found to be corrupted (contains no signal) due to an error in format conversion.

- **train_curated.csv** - ground truth labels for the curated subset of the training audio files (see Data Fields below)
- **train_noisy.csv** - ground truth labels for the noisy subset of the training audio files (see Data Fields below)
- **sample_submission.csv** - a sample submission file in the correct format, including the correct sorting of the sound categories; it contains the list of audio files found in the test.zip folder (corresponding to the public leaderboard)
- **train_curated.zip** - a folder containing the audio (.wav) training files of the curated subset
- **train_noisy.zip** - a folder containing the audio (.wav) training files of the noisy subset
- **test.zip** - a folder containing the audio (.wav) test files for the public leaderboard

### Columns
Each row of the train_curated.csv and train_noisy.csv files contains the following information:

- **fname**: the audio file name, eg, `0006ae4e.wav`
- **labels**: the audio classification label(s) (ground truth). Note that the number of labels per clip can be one, eg, `Bark` or more, eg, `"Walk_and_footsteps,Slam"`.

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (276 lines)
            sample_submission.csv (3362 lines)
            sample_submission.csv.zip (20.7 kB)
            test.zip (2.2 GB)
            train_curated.csv (4971 lines)
            train_curated.csv.zip (39.3 kB)
            train_curated.zip (2.4 GB)
            train_noisy.csv (19816 lines)
            train_noisy.csv.zip (154.2 kB)
            train_noisy.zip (21.5 GB)
            freesound-audio-tagging-2019/
                description.md (276 lines)
                sample_submission.csv (3362 lines)
                ... and 8 other files
                freesound-audio-tagging-2019/
                test/
                    4260ebea.wav (1.0 MB)
                    426eb1e0.wav (654.5 kB)
                    ... and 3359 other files
                    test/
                train_curated/
                    0006ae4e.wav (621.0 kB)
                    0019ef41.wav (181.3 kB)
                    ... and 4968 other files
                train_noisy/
                    00097e21.wav (1.3 MB)
                    000b6cfb.wav (1.3 MB)
                    ... and 19813 other files
            test/
                4260ebea.wav (1.0 MB)
                426eb1e0.wav (654.5 kB)
                ... and 3359 other files
                test/
            train_curated/
                0006ae4e.wav (621.0 kB)
                0019ef41.wav (181.3 kB)
                ... and 4968 other files
            train_noisy/
                00097e21.wav (1.3 MB)
                000b6cfb.wav (1.3 MB)
                ... and 19813 other files
        input/
            description.md (276 lines)
            sample_submission.csv (3362 lines)
            sample_submission.csv.zip (20.7 kB)
            test.zip (2.2 GB)
            train_curated.csv (4971 lines)
            train_curated.csv.zip (39.3 kB)
            train_curated.zip (2.4 GB)
            train_noisy.csv (19816 lines)
            train_noisy.csv.zip (154.2 kB)
            train_noisy.zip (21.5 GB)
            freesound-audio-tagging-2019/
                description.md (276 lines)
                sample_submission.csv (3362 lines)
                ... and 8 other files
                freesound-audio-tagging-2019/
                test/
                    4260ebea.wav (1.0 MB)
                    426eb1e0.wav (654.5 kB)
                    ... and 3359 other files
                    test/
                train_curated/
                    0006ae4e.wav (621.0 kB)
                    0019ef41.wav (181.3 kB)
                    ... and 4968 other files
                train_noisy/
                    00097e21.wav (1.3 MB)
                    000b6cfb.wav (1.3 MB)
                    ... and 19813 other files
            test/
                4260ebea.wav (1.0 MB)
                426eb1e0.wav (654.5 kB)
                ... and 3359 other files
                test/
                    4260ebea.wav (1.0 MB)
                    426eb1e0.wav (654.5 kB)
                    ... and 3359 other files
                    test/
            train_curated/
                0006ae4e.wav (621.0 kB)
                0019ef41.wav (181.3 kB)
                ... and 4968 other files
            train_noisy/
                00097e21.wav (1.3 MB)
                000b6cfb.wav (1.3 MB)
                ... and 19813 other files
        working/
            freesound-audio-tagging-2019/
                description.md (276 lines)
                sample_submission.csv (3362 lines)
                ... and 8 other files
                freesound-audio-tagging-2019/
                test/
                    4260ebea.wav (1.0 MB)
                    426eb1e0.wav (654.5 kB)
                    ... and 3359 other files
                    test/
                train_curated/
                    0006ae4e.wav (621.0 kB)
                    0019ef41.wav (181.3 kB)
                    ... and 4968 other files
                train_noisy/
                    00097e21.wav (1.3 MB)
                    000b6cfb.wav (1.3 MB)
                    ... and 19813 other files
```

-> data/freesound-audio-tagging-2019/sample_submission.csv has 3361 rows and 81 columns.
The columns are: fname, Accelerating_and_revving_and_vroom, Accordion, Acoustic_guitar, Applause, Bark, Bass_drum, Bass_guitar, Bathtub_(filling_or_washing), Bicycle_bell, Burping_and_eructation, Bus, Buzz, Car_passing_by, Cheering... and 66 more columns

-> data/freesound-audio-tagging-2019/train_curated.csv has 4970 rows and 2 columns.
The columns are: fname, labels

-> data/freesound-audio-tagging-2019/train_noisy.csv has 19815 rows and 2 columns.
The columns are: fname, labels

-> data/sample_submission.csv has 3361 rows and 81 columns.
The columns are: fname, Accelerating_and_revving_and_vroom, Accordion, Acoustic_guitar, Applause, Bark, Bass_drum, Bass_guitar, Bathtub_(filling_or_washing), Bicycle_bell, Burping_and_eructation, Bus, Buzz, Car_passing_by, Cheering... and 66 more columns

-> data/train_curated.csv has 4970 rows and 2 columns.
The columns are: fname, labels

-> data/train_noisy.csv has 19815 rows and 2 columns.
The columns are: fname, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.03737

# 6. Current score

0.02523

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.02498) has done: 'I remove the missing `lwlwrap` dependency and replace it with a small in-notebook implementation of label-weighted LRAP (used only for local sanity checks). Then I fix the import/NameError cascade by ensuring `MultiLabelBinarizer` is imported and cells run in order. Finally, I make submission generation robust by aligning the predicted label columns exactly to `sample_submission.csv` (both names and order), filling any missing classes with 0, and writing a valid `submission.csv` file.'
- What this solution (achieved 0.02498) has done: 'You’re currently submitting a constant per-class prior computed only from `train_curated`, which is very weak for lwlrap; the smallest legitimate improvement (without changing the “predict per-class constant probabilities” core logic) is to estimate that prior from more data. I keep the same approach but compute the per-class frequencies from `train_curated + train_noisy` combined, and I align the class set to the submission labels (fit the binarizer on `sample_submission` classes) so there’s no class-order mismatch. To reduce the domain shift from noisy labels while still using them, I add a tiny amount of label-smoothing and blend curated/noisy priors with a conservative weight (curated heavier), which typically lifts constant-baseline lwlrap toward your target without changing the modeling paradigm. The submission writing and column order remain exactly tied to `sample_submission.csv`.'
- What this solution (achieved 0.02523) has done: 'Your current approach is a constant-per-class prior; to move the score up toward the target without changing that core logic, the biggest safe gain is to estimate those priors in a label-weighted way that better matches lwlrap. I keep the same “predict a constant probability per class for all clips” paradigm, but compute per-class priors from curated+noisy using inverse-frequency reweighting so rare classes (which lwlrap weights equally) aren’t dominated by common ones. I also (a) drop known-bad curated files listed in the competition description to reduce noise in the prior estimate and (b) tune the curated/noisy blend slightly toward curated, which should improve while staying minimal. Submission column order and file writing remain identical to `sample_submission.csv`.'
- What this solution (achieved 0.02523) has done: 'I keep your constant-per-class prior approach intact and make only tiny, metric-aligned tweaks to move the score up toward your target. Specifically, I (1) adjust the curated/noisy blend slightly more toward curated (less label noise), and (2) make the “rare-label boosting” reweighting a bit gentler (so probabilities aren’t over-flattened), while keeping the same reweighted-prior computation. I also switch the smoothing from a fixed epsilon to a Jeffreys-style Beta prior (adds a tiny, principled amount of smoothing that usually improves ranking stability without changing semantics). Submission column alignment and file writing remain exactly tied to `sample_submission.csv`.'
- What this solution (achieved 0.02523) has done: 'To move your score upward toward the 0.03737 target while keeping the “constant per-class prior for every test clip” core logic, the smallest likely gain is to make the prior estimate more robust and less distorted by the current rare-class reweighting. I keep the same pipeline but (1) compute the reweighting using a gentle exponent (so it doesn’t over-flatten probabilities), (2) blend *both* the plain empirical prior and the reweighted prior (so common classes aren’t overly suppressed), and (3) tune the curated/noisy blend slightly more toward curated to reduce label noise. These are minimal, metric-aligned calibration changes that often improve label-weighted ranking metrics without changing the modeling paradigm or submission format. The script still writes a valid `submission.csv` aligned exactly to `sample_submission.csv`.'
- What this solution (achieved 0.02523) has done: 'You’re currently using a constant per-class prior, so the only safe way to move the lwlrap upward without changing core logic is to calibrate those priors to better match label-weighted behavior. I keep the same pipeline but (1) compute a label-balanced mixture weight between curated and noisy (per class) so rare classes aren’t overly dominated by noisy scarcity, and (2) apply a very small temperature sharpening to increase between-class score separation (helps ranking metrics) while keeping probabilities valid. I also keep the existing bad-file removal and submission column alignment exactly as in `sample_submission.csv`. These are minimal, metric-aligned adjustments that should increase the score toward your target without changing the “constant per-class probability for all test clips” approach.'
- What this solution (achieved 0.02523) has done: 'Your current constant-per-class prior baseline is already stable, so the smallest legitimate way to raise lwlrap toward the 0.03737 target is to slightly increase between-class score separation (ranking metric) without changing the model paradigm. I keep your exact pipeline but make the temperature sharpening a bit stronger and add a tiny “dynamic-range” expansion around 0.5 that preserves ordering while improving score spread. I also compute the effective sample size for the Beta smoothing from the actual binarized label matrix (so smoothing matches the label evidence rather than row count), which is a minimal calibration fix. Submission column order, class alignment, and CSV writing remain exactly tied to `sample_submission.csv`.'
- What this solution (achieved 0.02523) has done: 'We keep your constant-per-class prior approach exactly as-is, but make two metric-aligned calibration tweaks that typically improve label-weighted ranking without changing semantics: (1) compute the Beta smoothing strength from a realistic “effective sample size” based on the number of clips (not total positive labels), so rare labels aren’t over-smoothed into near-constants, and (2) very slightly increase score separation via a tiny adjustment to the temperature (sharpening) while keeping probabilities valid. These are minimal changes that should increase lwlrap from your current 0.02523 toward the 0.03737 target without altering architecture/training (there is none) or feature extraction (none). Submission formatting and class alignment remain strictly tied to `sample_submission.csv`, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.02523) has done: 'To move your lwlrap up toward 0.03737 while keeping the exact “constant per-class probability for every test clip” core logic, I only adjust calibration steps that affect ranking under a constant baseline. Specifically, I (1) make the Jeffreys/Beta smoothing use a per-class effective sample size (prevents rare labels from being over-smoothed toward a common constant), and (2) slightly strengthen the temperature sharpening to increase between-class separation (ranking metrics like lwlrap benefit) while leaving the model paradigm unchanged. Everything else—data loading, bad-file removal, prior/reweighted mixing, and exact submission column alignment—stays the same, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.02523) has done: 'We keep your “constant per-class probabilities for all test clips” baseline intact and only make small, metric-aligned calibration tweaks to nudge lwlrap upward toward 0.03737. The current pipeline likely over-flattens/over-distorts class priors via hard clipping after reweighting and the nonlinearity pair (sharpen + expand), so we (1) renormalize the reweighted prior to preserve the global positive rate (less distortion while keeping rare-class boost), and (2) slightly strengthen between-class separation with a tiny temperature tweak while backing off the expansion gamma to reduce over-warping. These are minimal, safe changes that preserve semantics (still one constant probability per class) and should improve ranking stability. Submission format/alignment and bad-file removal remain unchanged, and the script still writes `submission.csv`.'
- What this solution (achieved 0.02523) has done: 'To move your lwlrap upward toward the 0.03737 target without changing the “constant per-class probability for all test clips” core logic, I make two minimal, metric-aligned calibration adjustments. First, I replace the current rare-class reweighting (which can over-flatten and distort class ordering) with a label-balanced prior computed as the average of per-class positive rates *per label* using the combined curated+noisy set; this better matches label-weighted behavior while still producing one constant per class. Second, I slightly soften the current sharpening/expansion (keep them, just reduce their strength) to avoid over-warping probabilities, which tends to help ranking stability for lwlrap under a constant baseline. Submission formatting, class alignment to `sample_submission.csv`, and bad-file removal remain unchanged, and the script still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer


def _one_sample_lrap(y_true_row, y_score_row):
    """Label-ranking average precision for one sample.
    y_true_row: binary array shape (C,)
    y_score_row: scores array shape (C,)
    """
    pos = np.flatnonzero(y_true_row > 0)
    if pos.size == 0:
        return 0.0

    order = np.argsort(-y_score_row, kind="mergesort")
    y_true_sorted = y_true_row[order]

    cumsum_true = np.cumsum(y_true_sorted)
    ranks = np.arange(1, y_true_sorted.size + 1)
    precision_at_k = cumsum_true / ranks

    pos_ranks = np.flatnonzero(y_true_sorted > 0)
    return float(np.mean(precision_at_k[pos_ranks]))


def calculate_overall_lwlrap(y_true, y_score):
    """Compute label-weighted lwlrap approximating the competition metric.
    This is for local diagnostics only; it doesn't affect the submission.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    assert y_true.shape == y_score.shape
    n_samples, n_classes = y_true.shape

    per_sample_lrap = np.array(
        [_one_sample_lrap(y_true[i], y_score[i]) for i in range(n_samples)],
        dtype=np.float64,
    )

    class_counts = y_true.sum(axis=0).astype(np.float64)
    total_pos = class_counts.sum()
    if total_pos == 0:
        return 0.0

    per_class_lwlrap = np.zeros(n_classes, dtype=np.float64)
    for i in range(n_samples):
        pos = np.flatnonzero(y_true[i] > 0)
        if pos.size == 0:
            continue
        per_class_lwlrap[pos] += per_sample_lrap[i] / pos.size

    per_class_lwlrap = np.divide(
        per_class_lwlrap,
        class_counts,
        out=np.zeros_like(per_class_lwlrap),
        where=class_counts > 0,
    )
    label_weights = class_counts / total_pos
    return float(np.sum(per_class_lwlrap * label_weights))




## === cell 1
curated_df = pd.read_csv("../input/freesound-audio-tagging-2019/train_curated.csv")
noisy_df = pd.read_csv("../input/freesound-audio-tagging-2019/train_noisy.csv")
sample_df = pd.read_csv("../input/freesound-audio-tagging-2019/sample_submission.csv")

df_cur = curated_df.copy()
df_noi = noisy_df.copy()

df_cur["labels"] = df_cur["labels"].astype(str)
df_noi["labels"] = df_noi["labels"].astype(str)

bad_curated = {
    "f76181c4.wav",
    "77b925c2.wav",
    "6a1f682a.wav",
    "c7db12aa.wav",
    "7752cc8a.wav",
    "1d44b0bd.wav",  # corrupted (no signal)
}
df_cur = df_cur[~df_cur["fname"].isin(bad_curated)].reset_index(drop=True)



## === cell 2
sub_cols = list(sample_df.columns)
assert (
    sub_cols[0] == "fname"
), "Unexpected submission format: first column must be fname"
target_classes = sub_cols[1:]

mlb = MultiLabelBinarizer(classes=target_classes)
mlb.fit([[]])  # initializes with provided classes

true_labels_cur = mlb.transform(df_cur["labels"].str.split(","))
true_labels_noi = mlb.transform(df_noi["labels"].str.split(","))

sanity_self = calculate_overall_lwlrap(true_labels_cur, true_labels_cur)
sanity_zeros = calculate_overall_lwlrap(true_labels_cur, np.zeros_like(true_labels_cur))
print("Sanity lwlrap(y,y) =", sanity_self)
print("Sanity lwlrap(y,0) =", sanity_zeros)
print("Num submission classes:", len(target_classes))
print("Curated rows used:", len(df_cur), "Noisy rows used:", len(df_noi))



## === cell 3
label_means_cur = np.mean(true_labels_cur, axis=0).astype(np.float64)
label_means_noi = np.mean(true_labels_noi, axis=0).astype(np.float64)

counts_cur = true_labels_cur.sum(axis=0).astype(np.float64)
counts_noi = true_labels_noi.sum(axis=0).astype(np.float64)

k = 30.0
alpha_base = 0.92
alpha_per_class = alpha_base + (1.0 - alpha_base) * (
    counts_cur / (counts_cur + counts_noi + k)
)
alpha_per_class = np.clip(alpha_per_class, 0.85, 0.98)

p_emp = alpha_per_class * label_means_cur + (1.0 - alpha_per_class) * label_means_noi

p_label_balanced = np.full_like(p_emp, float(np.mean(p_emp)), dtype=np.float64)
mix_bal = 0.22  # small; keeps core logic but nudges toward label-weighted behavior
label_means = (1.0 - mix_bal) * p_emp + mix_bal * p_label_balanced

a = 0.5  # Jeffreys prior
n_eff_per_class = (counts_cur + counts_noi).astype(np.float64)
label_means = (label_means * n_eff_per_class + a) / (n_eff_per_class + 2.0 * a)
label_means = np.clip(label_means, 0.0, 1.0)


def _sharpen_probs(p, t=0.82, eps=1e-7):
    p = np.clip(p, eps, 1.0 - eps)
    logit = np.log(p / (1.0 - p))
    logit = logit / t
    return 1.0 / (1.0 + np.exp(-logit))


def _expand_around_half(p, gamma=1.10):
    p = np.asarray(p, dtype=np.float64)
    z = p - 0.5
    out = 0.5 + np.sign(z) * (np.abs(z) ** gamma)
    return np.clip(out, 0.0, 1.0)


label_means = _sharpen_probs(label_means, t=0.80)
label_means = _expand_around_half(label_means, gamma=1.03)

predicted_labels_train = np.repeat(
    label_means.reshape(1, -1), repeats=len(df_cur), axis=0
)
sanity_prior = calculate_overall_lwlrap(true_labels_cur, predicted_labels_train)
print("Sanity lwlrap(curated y, blended prior) =", sanity_prior)



## === cell 4
submission_array = np.repeat(label_means.reshape(1, -1), repeats=len(sample_df), axis=0)

submission = pd.DataFrame(submission_array, columns=target_classes)
submission.insert(0, "fname", sample_df["fname"].values)

submission = submission[sub_cols]

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print("Submission shape:", submission.shape)
print(submission.head())
