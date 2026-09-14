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

- What this solution (achieved 0.02498) has done: 'The crash happens immediately because the notebook imports `calculate_overall_lwlrap_sklearn` from a non-existent third-party module `lwlwrap`, which is not installed in this environment. To keep the notebook’s intended evaluation semantics, I replace that import with a small local implementation of `calculate_overall_lwlrap_sklearn` directly in the same cell. This preserves the same function name/signature so later cells can call it unchanged. No other logic (data loading, model code, etc.) is altered.'
- What this solution (achieved 0.02498) has done: 'Your current baseline trains no model and predicts per-label prevalence from only the curated set; the smallest legitimate way to move lwlrap upward toward 0.03737 is to compute those same prevalence predictions from more (still label-only) training data. I keep the exact same “predict label means for every clip” core logic, but include `train_noisy.csv` in the label-mean estimate and align the class order to `sample_submission.csv` to avoid any silent column/order mismatch. I also add light clipping of probabilities into \[0,1\] to keep the submission numerically well-formed without changing semantics. The output remains a valid `submission.csv` with the required header and ordering.'
- What this solution (achieved 0.02523) has done: 'I keep your “predict global label prevalence for every clip” baseline exactly the same, but reduce a known source of label noise that hurts the prevalence estimate: the 5 curated files with wrong labels and the 1 corrupted curated file mentioned in the competition description. This is a minimal, label-only data-cleaning step (no model/feature/training changes) that typically nudges lwlrap upward because the constant predictor becomes slightly better calibrated to true label frequencies. I apply the removal only to the curated metadata before concatenating with noisy, preserving the rest of your pipeline and the submission column order. The output remains `submission.csv` with the exact sample_submission schema.'
- What this solution (achieved 0.02523) has done: 'Your current “global label prevalence” predictor is already very constrained, so the smallest safe way to move lwlrap upward is to make the prevalence estimate less biased toward the (much larger) noisy set without changing the model form. I keep the exact same constant-per-class prediction, but compute the label means as a weighted average of curated and noisy prevalence (a light calibration step), while still using the same label binarization and submission column order. I also reuse the same curated cleanup you already applied, and keep probability clipping to \[0,1\]. This typically nudges the constant predictor toward better label ranking on the curated-like test distribution, improving score toward your 0.03737 target without changing core semantics.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import label_ranking_average_precision_score
from sklearn.preprocessing import MultiLabelBinarizer


def calculate_overall_lwlrap_sklearn(y_true, y_score):
    """
    Compute label-weighted label-ranking average precision (lwlrap) using sklearn.

    Parameters
    ----------
    y_true : array-like of shape (n_samples, n_classes)
        Binary indicator matrix of ground-truth labels.
    y_score : array-like of shape (n_samples, n_classes)
        Predicted scores/probabilities.

    Returns
    -------
    float
        Overall lwlrap score.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    y_score = np.nan_to_num(
        y_score,
        nan=-np.inf,
        posinf=np.finfo(np.float32).max,
        neginf=-np.finfo(np.float32).max,
    )

    per_sample_lrap = label_ranking_average_precision_score(
        y_true, y_score, sample_weight=None
    )

    if np.isscalar(per_sample_lrap):
        per_sample_lrap = np.array([per_sample_lrap], dtype=float)

    labels_per_sample = y_true.sum(axis=1).astype(float)
    total_labels = labels_per_sample.sum()
    if total_labels == 0:
        return 0.0

    return float(np.sum(per_sample_lrap * labels_per_sample) / total_labels)




## === cell 1
curated_df = pd.read_csv("../input/freesound-audio-tagging-2019/train_curated.csv")
noisy_df = pd.read_csv("../input/freesound-audio-tagging-2019/train_noisy.csv")

sample_df = pd.read_csv("../input/freesound-audio-tagging-2019/sample_submission.csv")

bad_curated_fnames = {
    "f76181c4.wav",
    "77b925c2.wav",
    "6a1f682a.wav",
    "c7db12aa.wav",
    "7752cc8a.wav",
    "1d44b0bd.wav",  # corrupted
}
curated_df = curated_df[~curated_df["fname"].isin(bad_curated_fnames)].reset_index(
    drop=True
)



## === cell 2
submission_classes = [c for c in sample_df.columns if c != "fname"]

mlb = MultiLabelBinarizer(classes=submission_classes)

true_labels_curated = mlb.fit_transform(curated_df["labels"].str.split(","))
true_labels_noisy = mlb.transform(noisy_df["labels"].str.split(","))

all_classes = mlb.classes_



## === cell 3
calculate_overall_lwlrap_sklearn(true_labels_curated, true_labels_curated)



## === cell 4
calculate_overall_lwlrap_sklearn(
    true_labels_curated, np.zeros_like(true_labels_curated)
)



## === cell 5
mean_curated = np.mean(true_labels_curated, axis=0)
mean_noisy = np.mean(true_labels_noisy, axis=0)

w_curated = 0.80
w_noisy = 1.0 - w_curated

label_means = (w_curated * mean_curated) + (w_noisy * mean_noisy)
label_means = np.clip(label_means, 0.0, 1.0)

df_all = pd.concat([curated_df, noisy_df], ignore_index=True)
true_labels_all = mlb.transform(df_all["labels"].str.split(","))
predicted_labels_all = np.repeat([label_means], len(df_all), axis=0)



## === cell 6
calculate_overall_lwlrap_sklearn(true_labels_all, predicted_labels_all)



## === cell 7
submission_labels = np.repeat([label_means], len(sample_df), axis=0)

submission = pd.DataFrame(submission_labels, columns=all_classes)

submission.insert(0, "fname", sample_df["fname"])
submission = submission[["fname"] + submission_classes]



## === cell 8
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(
    "Columns match sample_submission:",
    list(submission.columns) == list(sample_df.columns),
)
