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

# 8. Previous improvement plans

- What this solution (achieved 0.02498) has done: 'I replace the missing `lwlwrap` import with a small local implementation of the LWL‑RAP metric, correctly import `MultiLabelBinarizer`, and restructure the notebook so each step runs in order and produces a valid `submission.csv`. The baseline model simply predicts the overall label frequencies, which is sufficient to reach the modest target score.'
- What this solution (achieved 0.02498) has done: 'I incorporate the noisy training set into the label‑frequency baseline, recomputing the overall label probabilities from the combined curated + noisy data. This modest data‑augmentation is a minimal change that should raise the LWL‑RAP score toward the target without altering the core modeling approach.'
- What this solution (achieved 0.02498) has done: 'I add a lightweight weight‑search that blends the label frequencies from the clean curated set and the noisy set. By scanning a small range of weights on the training data and keeping the weight that maximizes the internal LWL‑RAP score, we keep the original frequency‑based model while nudging the predictions toward a better score. The rest of the pipeline (loading data, building the submission file) remains unchanged, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.02498) has done: 'I keep the overall frequency‑based model but add a lightweight “exponent scaling” step that boosts rarer labels (which are equally weighted in the competition). After finding the best curated/noisy blending weight, the code now scans a few exponent values, selects the one that maximizes the internal LRAP approximation, and uses that scaled label‑frequency vector for the final predictions. This small change should raise the validation LRAP and move the public score closer to the target while preserving the original logic.'
- What this solution (achieved 0.02498) has done: 'I keep the original frequency‑based approach but improve the label probability estimation by applying Laplace smoothing (to avoid zeros) and by searching a slightly finer grid of blending weights and exponent values. These small, targeted tweaks are expected to raise the LWL‑RAP score toward the target without changing the overall model logic.'
- What this solution (achieved 0.02498) has done: 'I keep the overall frequency‑based approach but improve the internal hyper‑parameter search so the model predicts a slightly better calibrated label distribution.  
- The blending weight between curated and noisy frequencies is now scanned at 0.01 steps (instead of 0.025).  
- The exponent used to boost rare labels is explored on a finer grid (0.1 → 1.0 in steps of 0.1).  
These small changes keep the core logic intact while nudging the training LRAP score upward, moving it closer to the target 0.03737.'
- What this solution (achieved 0.02498) has done: 'I keep the overall frequency‑based approach but make two tiny, targeted tweaks that are expected to raise the label‑ranking‑average‑precision without changing the core logic: (1) use a smaller Laplace smoothing constant (0.5) when estimating the per‑label frequencies, and (2) extend the exponent search up to 2.0 (with finer steps) so rarer labels can be boosted more aggressively. These adjustments keep the same model structure while nudging the predictions toward the target score.'
- What this solution (achieved 0.02498) has done: 'I reduce the Laplace smoothing constant (from 0.5 to 0.1) to give rarer labels higher raw frequencies and expand the exponent search up to 4.0 (denser grid). This keeps the same frequency‑based model but should boost the internal LWL‑RAP score, moving it closer to the target without altering the overall pipeline.'
- What this solution (achieved 0.02498) has done: 'I add a hold‑out validation split so the blending weight `w` and exponent `α` are chosen based on a validation LRAP rather than the full training set, and I tighten the smoothing to 0.01 and expand the exponent search up to 8.0. These minimal tweaks keep the frequency‑based model unchanged while giving a better‑calibrated label distribution that should raise the LWL‑RAP score toward the target.'
- What this solution (achieved 0.02498) has done: 'I lower the Laplace smoothing to a very small value and select the blending weight `w` and exponent `α` by maximizing the LRAP on the full training set (instead of a hold‑out split). This keeps the frequency‑based model intact while giving a modest boost to the LWL‑RAP score, moving it closer to the target.'
- What this solution (achieved 0.02498) has done: 'I keep the overall frequency‑based model but improve its calibration by using a stronger Laplace smoothing ( 0.5 ) and selecting the blending weight and exponent on a held‑out validation split instead of the full training set. This modest change respects the original logic while expectedly raising the LWL‑RAP score toward the target.'
- What this solution (achieved 0.02498) has done: 'I lower the Laplace smoothing to 0.01 (so rare labels get higher raw frequencies), search the curated‑noisy blending weight on a finer 0.01 grid, and extend the exponent scaling search up to 30 with 0.1 steps. These small, targeted tweaks keep the frequency‑based model intact while likely raising the validation LWL‑RAP and moving the score closer to the target.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.model_selection import train_test_split


def calculate_overall_lwlrap_sklearn(y_true, y_pred):
    """
    Wrapper that returns LRAP using scikit‑learn as an approximation
    of the competition’s LWL‑RAP metric.
    """
    return label_ranking_average_precision_score(y_true, y_pred)


curated_df = pd.read_csv("../input/freesound-audio-tagging-2019/train_curated.csv")
noisy_df = pd.read_csv("../input/freesound-audio-tagging-2019/train_noisy.csv")
df = pd.concat([curated_df, noisy_df], ignore_index=True)

sample_df = pd.read_csv("../input/freesound-audio-tagging-2019/sample_submission.csv")

mlb = MultiLabelBinarizer()
true_labels = mlb.fit_transform(df["labels"].str.split(","))  # (n_samples, n_labels)
all_classes = mlb.classes_

curated_true = mlb.transform(curated_df["labels"].str.split(","))
noisy_true = mlb.transform(noisy_df["labels"].str.split(","))

n_labels = len(all_classes)

smoothing = 0.01

curated_smoothed = (np.sum(curated_true, axis=0) + smoothing) / (
    curated_true.shape[0] + n_labels * smoothing
)
noisy_smoothed = (np.sum(noisy_true, axis=0) + smoothing) / (
    noisy_true.shape[0] + n_labels * smoothing
)

indices = np.arange(len(df))
train_idx, val_idx = train_test_split(
    indices, test_size=0.1, random_state=42
)  # 10 % hold‑out

val_true = true_labels[val_idx]

best_w = 0.0
best_score = -1.0
for w in np.linspace(0, 1, 101):  # step = 0.01
    label_means_w = w * curated_smoothed + (1 - w) * noisy_smoothed
    pred_full = np.tile(label_means_w, (len(df), 1))
    pred_val = pred_full[val_idx]
    score = calculate_overall_lwlrap_sklearn(val_true, pred_val)
    if score > best_score:
        best_score = score
        best_w = w

label_means = best_w * curated_smoothed + (1 - best_w) * noisy_smoothed

best_alpha = 1.0
best_score_alpha = -1.0
for a in np.arange(0.1, 30.1, 0.1):  # step = 0.1
    scaled_means = np.power(label_means, a)
    pred_full = np.tile(scaled_means, (len(df), 1))
    pred_val = pred_full[val_idx]
    score = calculate_overall_lwlrap_sklearn(val_true, pred_val)
    if score > best_score_alpha:
        best_score_alpha = score
        best_alpha = a

final_label_means = np.power(label_means, best_alpha)

print(
    f"Best blending weight (curated): {best_w:.3f}, "
    f"Best exponent: {best_alpha:.2f}, "
    f"Validation LWL‑RAP (approx): {best_score_alpha:.5f}"
)



## === cell 1
predicted_labels = np.tile(final_label_means, (len(df), 1))
train_lwlrap = calculate_overall_lwlrap_sklearn(true_labels, predicted_labels)
print(f"Training LWL‑RAP with blended & scaled frequencies: {train_lwlrap:.5f}")



## === cell 2
ordered_label_means = (
    pd.Series(final_label_means, index=all_classes)
    .reindex(sample_df.columns[1:])  # skip 'fname'
    .fillna(0)
    .values
)

submission_matrix = np.tile(ordered_label_means, (len(sample_df), 1))



## === cell 3
submission = pd.DataFrame(submission_matrix, columns=sample_df.columns[1:])
submission.insert(0, "fname", sample_df["fname"])
submission.to_csv("submission.csv", index=False)
print("submission.csv written – ready for upload.")
