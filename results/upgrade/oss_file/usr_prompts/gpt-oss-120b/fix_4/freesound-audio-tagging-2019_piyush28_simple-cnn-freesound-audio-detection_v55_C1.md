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

No external packages required in the script and installed.

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

0.3601589627320324

# 6. Current score

0.0328

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.02498) has done: 'I replace the broken TensorFlow pipeline with a simple, fully‑compatible baseline: load the curated and noisy training CSVs, compute the empirical frequency of each tag, and fill the test submission with those frequencies (the same probability for every file). This eliminates the import errors, missing files, and TF‑1‑only code while still using the original label set, producing a valid `submission.csv` that can be scored (the frequency baseline typically approaches the target region). The implementation follows the original cell structure but only contains the necessary, safe operations.'
- What this solution (achieved 0.02498) has done: 'I keep the overall frequency‑based baseline but compute separate per‑label frequencies for the curated and noisy subsets and combine them with a higher weight for the curated data (which is higher‑quality). This small change preserves the original logic while providing a modest, expected lift in LRAP, moving the score closer to the target. The rest of the pipeline (reading CSVs, building the submission DataFrame and writing it) remains unchanged.'
- What this solution (achieved 0.0328) has done: 'I keep the overall frequency‑based approach but concentrate the predictions on the most common tags. After computing the weighted label frequencies I select the top K labels (K = 10) and set their probability to 1.0 for every test file while all other labels receive 0.0. This retains the original logic, creates a valid submission, and should raise the LRAP score substantially toward the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

possible_roots = ["../input/freesound-audio-tagging-2019", "../input"]
root_dir = None
for p in possible_roots:
    if os.path.isdir(p):
        root_dir = p
        break
if root_dir is None:
    raise FileNotFoundError("Could not locate input data directory.")

train_curated_csv = os.path.join(root_dir, "train_curated.csv")
train_noisy_csv = os.path.join(root_dir, "train_noisy.csv")
sample_sub_csv = os.path.join(root_dir, "sample_submission.csv")



## === cell 1
train_curated_df = pd.read_csv(train_curated_csv)
train_noisy_df = pd.read_csv(train_noisy_csv)

train_df = pd.concat([train_curated_df, train_noisy_df], ignore_index=True)

label_lists = (
    train_df["labels"].fillna("").apply(lambda x: [lbl for lbl in x.split(",") if lbl])
)

all_labels = sorted({lbl for sublist in label_lists for lbl in sublist})
mlb = MultiLabelBinarizer(classes=all_labels)

curated_labels = (
    train_curated_df["labels"]
    .fillna("")
    .apply(lambda x: [lbl for lbl in x.split(",") if lbl])
)
curated_binary = mlb.fit_transform(curated_labels)
curated_freq = curated_binary.mean(axis=0)  # (n_labels,)

noisy_labels = (
    train_noisy_df["labels"]
    .fillna("")
    .apply(lambda x: [lbl for lbl in x.split(",") if lbl])
)
noisy_binary = mlb.transform(noisy_labels)
noisy_freq = noisy_binary.mean(axis=0)

weight_curated = 0.7
weight_noisy = 0.3
label_frequencies = weight_curated * curated_freq + weight_noisy * noisy_freq

freq_dict = dict(zip(mlb.classes_, label_frequencies))

top_k = 10  # number of most common tags to set to probability 1.0
top_labels = [
    label
    for label, _ in sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
]



## === cell 2
sample_sub = pd.read_csv(sample_sub_csv)
submission_columns = list(sample_sub.columns)  # ['fname', label1, label2, ...]
label_columns = submission_columns[1:]  # all label names, already in correct order



## === cell 3
pred_dict = {"fname": sample_sub["fname"].tolist()}
for label in label_columns:
    prob = 1.0 if label in top_labels else 0.0
    pred_dict[label] = [prob] * len(pred_dict["fname"])



## === cell 4
submission_df = pd.DataFrame(pred_dict, columns=submission_columns)
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path} with shape {submission_df.shape}")
