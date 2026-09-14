# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.7

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import label_ranking_average_precision_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import StratifiedKFold


def calculate_overall_lwlrap_sklearn(y_true, y_score):
    """
    Compute label-weighted label-ranking average precision (lwlrap).

    Score-relevant FIX:
    - The previous implementation attempted to compute per-label LRAP by passing
      a single-column matrix into sklearn's label_ranking_average_precision_score.
      With only 1 label, LRAP is degenerate (always 1.0 when that label is present),
      so CV tuning becomes meaningless and can hurt leaderboard score.

    Correct lwlrap definition for this competition:
      lwlrap = mean over labels c that are present in the set of:
                 LRAP computed over samples, but considering the *full* label ranking,
                 and then averaging the per-sample contributions only for samples
                 where label c is actually present.

    Implementation:
    - Compute per-sample LRAP contributions over all labels (sklearn does this).
    - Convert to per-label scores by averaging those per-sample LRAP values over
      samples where each label is positive.
    - Finally average across labels with at least one positive.
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
        y_true, y_score, average=None
    )
    per_sample_lrap = np.asarray(per_sample_lrap, dtype=np.float64)

    label_pos = (y_true > 0).astype(np.float64)
    label_counts = label_pos.sum(axis=0)  # (n_classes,)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_label_scores = (label_pos * per_sample_lrap[:, None]).sum(
            axis=0
        ) / label_counts

    present_mask = label_counts > 0
    if not np.any(present_mask):
        return 0.0
    return float(np.mean(per_label_scores[present_mask]))


def apply_probability_temperature(p, alpha):
    """
    Minimal, monotonic calibration of per-class probabilities.
    We use a power transform: p' = p**alpha (alpha > 0).
    """
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 0.0, 1.0)
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    return np.clip(p ** float(alpha), 0.0, 1.0)


def _foldwise_label_weighted_mean(scores, y_true_fold):
    """
    Score-relevant: in curated CV, different folds can contain different sets and
    counts of positive labels. Since the competition weights labels equally, we
    reduce CV noise/bias by weighting each fold score by how many labels are
    actually present in that fold (proxy for "label coverage").
    """
    y_true_fold = np.asarray(y_true_fold)
    label_present = (y_true_fold.sum(axis=0) > 0).astype(np.float64)
    w = float(label_present.sum())
    return float(scores), w


def mean_with_laplace_smoothing(y_bin, smoothing):
    """
    Score-relevant minimal change:
    For a constant-per-class predictor, raw prevalence can under-estimate rare
    labels; symmetric Laplace smoothing (add-k) nudges probabilities away from 0/1
    while preserving per-class ranking monotonicity across classes reasonably.

    mean = (pos + k) / (n + 2k)
    """
    y_bin = np.asarray(y_bin)
    n = float(y_bin.shape[0])
    pos = np.sum(y_bin, axis=0).astype(np.float64)
    k = float(smoothing)
    return (pos + k) / (n + 2.0 * k)




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
def calculate_overall_lwlrap_sklearn(y_true, y_score):
    """
    Compute label-weighted label-ranking average precision (lwlrap).

    Score-relevant FIX:
    - The previous implementation attempted to compute per-label LRAP by passing
      a single-column matrix into sklearn's label_ranking_average_precision_score.
      With only 1 label, LRAP is degenerate (always 1.0 when that label is present),
      so CV tuning becomes meaningless and can hurt leaderboard score.

    Correct lwlrap definition for this competition:
      lwlrap = mean over labels c that are present in the set of:
                 LRAP computed over samples, but considering the *full* label ranking,
                 and then averaging the per-sample contributions only for samples
                 where label c is actually present.

    Implementation:
    - Compute per-sample LRAP contributions over all labels (sklearn does this).
    - Convert to per-label scores by averaging those per-sample LRAP values over
      samples where each label is positive.
    - Finally average across labels with at least one positive.
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
        y_true, y_score, average=None
    )
    per_sample_lrap = np.asarray(per_sample_lrap, dtype=np.float64)

    label_pos = (y_true > 0).astype(np.float64)
    label_counts = label_pos.sum(axis=0)  # (n_classes,)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_label_scores = (label_pos * per_sample_lrap[:, None]).sum(
            axis=0
        ) / label_counts

    present_mask = label_counts > 0
    if not np.any(present_mask):
        return 0.0
    return float(np.mean(per_label_scores[present_mask]))


calculate_overall_lwlrap_sklearn(true_labels_curated, true_labels_curated)


## --- ERROR in cell 3, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3120798705.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     51[0m [0;34m[0m[0m
[1;32m     52[0m [0;34m[0m[0m
[0;32m---> 53[0;31m [0mcalculate_overall_lwlrap_sklearn[0m[0;34m([0m[0mtrue_labels_curated[0m[0;34m,[0m [0mtrue_labels_curated[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/tmp/ipykernel_11/3120798705.py[0m in [0;36mcalculate_overall_lwlrap_sklearn[0;34m(y_true, y_score)[0m
[1;32m     33[0m     [0;31m# Bug fix: request per-sample LRAP values; the default "macro" returns a scalar,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     34[0m     [0;31m# which then crashes when indexed as if it were (n_samples,).[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 35[0;31m     per_sample_lrap = label_ranking_average_precision_score(
[0m[1;32m     36[0m         [0my_true[0m[0;34m,[0m [0my_score[0m[0;34m,[0m [0maverage[0m[0;34m=[0m[0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[1;32m     37[0m     )

[0;31mTypeError[0m: label_ranking_average_precision_score() got an unexpected keyword argument 'average'

## === cell 4
calculate_overall_lwlrap_sklearn(
    true_labels_curated, np.zeros_like(true_labels_curated)
)
