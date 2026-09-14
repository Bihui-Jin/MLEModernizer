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

0.02498

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.02498) has done: 'I remove the missing `lwlwrap` dependency and replace it with a small in-notebook implementation of label-weighted LRAP (used only for local sanity checks). Then I fix the import/NameError cascade by ensuring `MultiLabelBinarizer` is imported and cells run in order. Finally, I make submission generation robust by aligning the predicted label columns exactly to `sample_submission.csv` (both names and order), filling any missing classes with 0, and writing a valid `submission.csv` file.'

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
sample_df = pd.read_csv("../input/freesound-audio-tagging-2019/sample_submission.csv")

df = pd.concat([curated_df], ignore_index=True)



## === cell 2
mlb = MultiLabelBinarizer()
true_labels = mlb.fit_transform(df["labels"].str.split(","))
train_classes = list(mlb.classes_)

sanity_self = calculate_overall_lwlrap(true_labels, true_labels)
sanity_zeros = calculate_overall_lwlrap(true_labels, np.zeros_like(true_labels))
print("Sanity lwlrap(y,y) =", sanity_self)
print("Sanity lwlrap(y,0) =", sanity_zeros)
print("Num train classes:", len(train_classes))



## === cell 3
label_means = np.mean(true_labels, axis=0).astype(np.float64)

predicted_labels_train = np.repeat(label_means.reshape(1, -1), repeats=len(df), axis=0)
sanity_prior = calculate_overall_lwlrap(true_labels, predicted_labels_train)
print("Sanity lwlrap(y, prior) =", sanity_prior)



## === cell 4
sub_cols = list(sample_df.columns)
assert (
    sub_cols[0] == "fname"
), "Unexpected submission format: first column must be fname"
target_classes = sub_cols[1:]

train_prior_map = {cls: label_means[i] for i, cls in enumerate(train_classes)}
submission_array = np.zeros((len(sample_df), len(target_classes)), dtype=np.float64)
for j, cls in enumerate(target_classes):
    submission_array[:, j] = train_prior_map.get(cls, 0.0)

submission = pd.DataFrame(submission_array, columns=target_classes)
submission.insert(0, "fname", sample_df["fname"].values)

submission = submission[sub_cols]



## === cell 5
out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print("Submission shape:", submission.shape)
print(submission.head())
