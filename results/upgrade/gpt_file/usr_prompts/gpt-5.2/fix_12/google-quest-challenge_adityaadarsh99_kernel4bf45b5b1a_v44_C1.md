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
Given questions and answers from various StackExchange properties, predict target values of 30 labels for each question-answer pair.

## Metric
Mean column-wise Spearman's correlation coefficient. The Spearman's rank correlation is computed for each target column, and the mean of these values is calculated for the submission score.

## Submission Format
For each qa_id in the test set, you must predict a probability for each target variable. The predictions should be in the range [0,1]. The file should contain a header and have the following format:

```
qa_id,question_asker_intent_understanding,...,answer_well_written
6,0.0,...,0.5
8,0.5,...,0.1
18,1.0,...,0.0
etc.
```

## Dataset
The list of 30 target labels are the same as the column names in the `sample_submission.csv` file. Target labels with the prefix `question_` relate to the `question_title` and/or `question_body` features in the data. Target labels with the prefix `answer_` relate to the `answer` feature.

Target labels are aggregated from multiple raters, and can have continuous values in the range `[0,1]`. Therefore, predictions must also be in that range.

- **train.csv** - the training data (target labels are the last 30 columns)
- **test.csv** - the test set (you must predict 30 labels for each test set row)
- **sample_submission.csv** - a sample submission file in the correct format; column names are the 30 target labels

# 2. Python version

3.8

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
            description.md (83 lines)
            sample_submission.csv (609 lines)
            sample_submission.csv.zip (8.9 kB)
            test.csv (19551 lines)
            test.csv.zip (471.7 kB)
            train.csv (159837 lines)
            train.csv.zip (4.2 MB)
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
        input/
            description.md (83 lines)
            sample_submission.csv (609 lines)
            sample_submission.csv.zip (8.9 kB)
            test.csv (19551 lines)
            test.csv.zip (471.7 kB)
            train.csv (159837 lines)
            train.csv.zip (4.2 MB)
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
        working/
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
```

-> data/google-quest-challenge/sample_submission.csv has 608 rows and 31 columns.
The columns are: qa_id, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer, question_fact_seeking, question_has_commonly_accepted_answer, question_interestingness_others, question_interestingness_self, question_multi_intent, question_not_really_a_question, question_opinion_seeking, question_type_choice, question_type_compare, question_type_consequence... and 16 more columns

-> data/google-quest-challenge/test.csv has 19550 rows and 11 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host

-> data/google-quest-challenge/train.csv has 159836 rows and 41 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer... and 26 more columns

-> data/sample_submission.csv has 608 rows and 31 columns.
The columns are: qa_id, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer, question_fact_seeking, question_has_commonly_accepted_answer, question_interestingness_others, question_interestingness_self, question_multi_intent, question_not_really_a_question, question_opinion_seeking, question_type_choice, question_type_compare, question_type_consequence... and 16 more columns

-> data/test.csv has 19550 rows and 11 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host

-> data/train.csv has 159836 rows and 41 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer... and 26 more columns

-> (stopped after 10 files for performance)

# 5. Target score

-0.0015378042945261

# 6. Current score

0.0272

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved nan) has done: 'I remove the dependency on a missing external file (`../input/bert-cnn/temp.csv`) and instead generate a valid baseline submission directly from the provided competition files. To keep the core intent (produce a submission) with minimal logic, the script load `sample_submission.csv`, ensure all required columns exist, fill predictions with a constant value in `[0,1]`, and write `submission.csv`. This fixes the runtime errors (undefined `sub`, `temp`, etc.) and guarantees an end-to-end run that outputs a correctly formatted CSV. Since no current score exists, this aims to produce a stable, valid (not necessarily strong) score while meeting the competition’s submission constraints.'
- What this solution (achieved -0.01016) has done: 'Your current pipeline already writes a valid `submission.csv`, but the constant-0.5 predictions can lead to unstable or even undefined (NaN) Spearman correlation in some implementations when a column has no variance. To move the score toward the target (and avoid NaNs), I keep the same “no-model baseline” core logic but replace the constant with deterministic, tiny per-row variation that stays in `[0,1]` and is the same for all target columns. This preserves the semantics (a simple baseline submission) while ensuring each column has non-zero variance, which makes Spearman well-defined and should yield a finite score. I also remove the unused `MinMaxScaler` import and keep `temp` inspection cells working unchanged.'
- What this solution (achieved -0.01016) has done: 'Your current baseline already avoids NaNs by adding tiny per-row variation, but the variation is identical across all 30 targets, which tends to make column-wise Spearman correlations uniformly weak. To move the score upward toward your target with minimal change and without introducing any modeling, I keep the same submission-building logic but generate a deterministic, very small *column-specific* variation per target (still within a narrow band around 0.5). This preserves the “no-model baseline” semantics while increasing the chance that at least some target columns align better with the true ranks. I also keep the exact required column order and ensure output stays in \[0,1\] and produces `submission.csv`.'
- What this solution (achieved -0.00094) has done: 'Your current submission is still essentially “near-constant with a tiny monotonic row ramp,” which is largely uncorrelated with the true ranks and can land around small negative Spearman. To move the score upward toward your (still slightly negative) target with minimal semantic change, I keep the same baseline approach but replace the generic row ramp with simple, deterministic text-derived signals from the provided `test.csv` fields (lengths of question title/body/answer). This remains non-ML, fast, and produces per-column variation that is more plausibly aligned with several targets than a pure index-based ramp. I also keep predictions clipped to `[0,1]`, preserve the sample-submission column order, and still write a valid `submission.csv`.'
- What this solution (achieved -8e-05) has done: 'Your current score (-0.00094) is better than the target (-0.0015378), so to move closer (while staying valid) we should slightly reduce performance with the smallest possible change. I keep the exact same feature signals and submission-building logic, but shrink the per-column variation amplitude so predictions are closer to a near-constant baseline (which tends to reduce absolute Spearman magnitude). This is a minimal, deterministic parameter tweak (no architecture/training/feature changes) and still guarantees non-constant columns and valid \[0,1\] outputs. The script still write a correct `submission.csv` end-to-end.'
- What this solution (achieved -0.02385) has done: 'Your current score (-8e-05) is already better than the target (-0.0015378), so to move closer we should very slightly *reduce* correlation strength with the smallest safe tweak. I keep the exact same feature signals and submission construction, and only shrink the per-column variation amplitude a bit more so predictions become closer to a near-constant baseline while still having non-zero variance (avoids Spearman NaNs). I also make the per-column hash deterministic across runs (Python’s built-in `hash()` is randomized per session), which stabilizes the score rather than accidentally drifting away from the target. The output schema, clipping to [0,1], and `submission.csv` writing remain unchanged.'
- What this solution (achieved 0.0271) has done: 'Your current score (-0.02385) is worse than the target (-0.0015378), so we should gently increase performance while keeping the same “fast heuristic, no-ML” core logic. The smallest high-signal change is to build the per-target prediction from a few deterministic, text-derived features that plausibly align with specific labels (e.g., “expect_short_answer” vs “fact_seeking”, or “answer_well_written” vs answer length), instead of using essentially random per-column weights. We keep the same feature family (length-based, robust z-scores, tanh squashing to [0,1]) but replace the per-column CRC weight mixing with a fixed, label-aware mapping for the most semantically obvious columns and a safe default for the rest. This should move Spearman correlations upward (less negative) without changing the overall approach or runtime, and still writes a valid `submission.csv`.'
- What this solution (achieved 0.0271) has done: 'Your current score (0.0271) is much higher than the target (-0.0015378), so we should *slightly reduce* performance to move closer while keeping the same heuristic, no-ML core logic. The smallest safe lever is to shrink the signal amplitude so predictions become closer to a near-constant baseline (which tends to reduce Spearman correlation magnitude) while still maintaining non-zero variance and valid \[0,1\] outputs. I only adjust `amp` (and keep all feature extraction, label-aware mapping, squashing, and submission formatting identical) to nudge the score downward toward the target band. This keeps runtime unchanged and preserves end-to-end CSV generation.'
- What this solution (achieved 0.0271) has done: 'Your current score (0.0271) is far above the target (-0.0015378), so we should intentionally (but safely) reduce correlation magnitude to move closer to the target without changing the overall heuristic approach. The smallest, most reliable lever is to reduce the signal amplitude `amp`, making predictions closer to a near-constant baseline while still preserving non-zero variance (to avoid Spearman NaNs). I keep the same feature extraction, label-aware weighting, tanh squashing, clipping, and submission formatting; only `amp` is adjusted. This should nudge the score downward toward the target tolerance band while remaining deterministic and valid.'
- What this solution (achieved 0.0271) has done: 'Your current score (0.0271) is far above the target (-0.0015378), so the goal is to reduce correlation magnitude with the smallest possible change while keeping the same fast heuristic and submission format. The most reliable lever is to shrink the signal amplitude `amp`, pushing predictions closer to a near-constant baseline (which tends to reduce Spearman), while keeping non-zero variance to avoid NaNs. I only adjust `amp` and leave feature extraction, label-aware weights, tanh squashing, clipping, and CSV writing unchanged. This should move the score downward toward the target band with minimal risk and identical core logic.'
- What this solution (achieved 0.0272) has done: 'Your current score (0.0271) is far above the target (-0.0015378), so we should intentionally reduce correlation strength while keeping the exact same heuristic feature extraction and label-aware weighting logic. The smallest reliable lever is to shrink the signal amplitude `amp`, making predictions closer to a near-constant baseline but still with non-zero variance (avoids Spearman NaNs). I only change `amp` and keep all other computations, column ordering, clipping to [0,1], and `submission.csv` writing identical. This should move the score downward toward the target band with minimal risk and within the runtime limit.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
SAMPLE_PATHS = [
    "/kaggle/input/google-quest-challenge/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
]
TEST_PATHS = [
    "/kaggle/input/google-quest-challenge/test.csv",
    "/kaggle/input/test.csv",
]


def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of the paths exist: {paths}")


sample_path = first_existing(SAMPLE_PATHS)
test_path = first_existing(TEST_PATHS)

sample_submission = pd.read_csv(sample_path)

test_df = pd.read_csv(
    test_path,
    usecols=["qa_id", "question_title", "question_body", "answer"],
)

target_cols = [c for c in sample_submission.columns if c != "qa_id"]



## === cell 2
sub = pd.DataFrame({"qa_id": test_df["qa_id"].values})

qt = test_df["question_title"].fillna("").astype(str)
qb = test_df["question_body"].fillna("").astype(str)
ans = test_df["answer"].fillna("").astype(str)

qt_len = qt.str.len().astype(np.float32).values
qb_len = qb.str.len().astype(np.float32).values
a_len = ans.str.len().astype(np.float32).values

q_marks = qb.str.count(r"\?").astype(np.float32).values
excl_marks = qb.str.count(r"!").astype(np.float32).values
a_lines = ans.str.count(r"\n").astype(np.float32).values


def robust_z(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    med = np.median(x).astype(np.float32)
    mad = np.median(np.abs(x - med)).astype(np.float32)
    scale = np.float32(1.4826) * mad + np.float32(1e-6)
    return ((x - med) / scale).astype(np.float32)


z_qt = robust_z(qt_len)
z_qb = robust_z(qb_len)
z_a = robust_z(a_len)
z_qm = robust_z(q_marks)
z_em = robust_z(excl_marks)
z_al = robust_z(a_lines)

detail = (
    np.float32(0.50) * z_qb + np.float32(0.20) * z_qt + np.float32(0.30) * z_a
).astype(np.float32)

base = np.float32(0.5)

amp = np.float32(0.00002)

weights = {}
for c in target_cols:
    w_detail, w_qt, w_qb, w_a, w_qm, w_em, w_al = (
        np.float32(1.0),
        np.float32(0.0),
        np.float32(0.0),
        np.float32(0.0),
        np.float32(0.0),
        np.float32(0.0),
        np.float32(0.0),
    )

    lc = c.lower()

    if lc.startswith("question_"):
        w_qb += np.float32(0.35)
        w_qt += np.float32(0.20)
        w_qm += np.float32(0.10)

        if "expect_short_answer" in lc:
            w_qb += np.float32(-0.50)
            w_qm += np.float32(0.10)
        if "fact_seeking" in lc:
            w_qm += np.float32(0.25)
            w_qb += np.float32(0.10)
        if "opinion_seeking" in lc:
            w_em += np.float32(0.15)
            w_qb += np.float32(0.10)
        if "conversational" in lc:
            w_em += np.float32(0.25)
            w_qt += np.float32(0.10)
        if "not_really_a_question" in lc:
            w_qm += np.float32(-0.35)
            w_qb += np.float32(-0.10)
        if "has_commonly_accepted_answer" in lc:
            w_qm += np.float32(0.15)
        if "multi_intent" in lc:
            w_qm += np.float32(0.15)
            w_qb += np.float32(0.15)
        if "type_choice" in lc:
            w_qm += np.float32(0.10)
        if "type_compare" in lc:
            w_qb += np.float32(0.15)
        if "type_consequence" in lc:
            w_qb += np.float32(0.10)
        if "interestingness_self" in lc or "interestingness_others" in lc:
            w_qb += np.float32(0.20)
            w_em += np.float32(0.10)

    if lc.startswith("answer_"):
        w_a += np.float32(0.55)
        w_al += np.float32(0.20)

        if "helpful" in lc:
            w_a += np.float32(0.20)
            w_al += np.float32(0.15)
        if "well_written" in lc:
            w_al += np.float32(0.25)
        if "relevance" in lc:
            w_a += np.float32(0.15)
        if "satisfaction" in lc:
            w_a += np.float32(0.10)

    weights[c] = (w_detail, w_qt, w_qb, w_a, w_qm, w_em, w_al)

for c in target_cols:
    w_detail, w_qt, w_qb, w_a, w_qm, w_em, w_al = weights[c]

    logits = amp * (
        w_detail * detail
        + w_qt * z_qt
        + w_qb * z_qb
        + w_a * z_a
        + w_qm * z_qm
        + w_em * z_em
        + w_al * z_al
    ).astype(np.float32)

    pred = (base + np.float32(0.25) * np.tanh(logits)).astype(np.float32)
    sub[c] = pred

sub = sub[["qa_id"] + target_cols]
sub[target_cols] = sub[target_cols].astype(np.float32).clip(0.0, 1.0)

sub.head()



## === cell 3
assert sub.shape[0] == test_df.shape[0], "Submission rows must match test rows."
assert (
    list(sub.columns) == ["qa_id"] + target_cols
), "Submission columns must match required format."
assert (
    sub[target_cols].min().min() >= 0.0 and sub[target_cols].max().max() <= 1.0
), "Predictions must be within [0,1]."

sub.describe(include="all").T.head()



## === cell 4
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)



## === cell 5
temp = sub.copy()



## === cell 6
"""
min_max = MinMaxScaler(feature_range=(0.01, 0.99))
temp.iloc[:,1:] = min_max.fit_transform(temp.iloc[:,1:])

temp.sort_values(by='qa_id',axis=0,inplace=True)
temp = temp.reset_index(drop=True)
temp.to_csv("submission.csv",index=False,float_format= '%.20f')"""
pass



## === cell 7
temp



## === cell 8
temp.iloc[:, 1:].min()



## === cell 9
temp.iloc[:, 1:].max()
