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

-0.01273

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved nan) has done: 'Your notebook fails because it tries to read a non-existent external dataset (`../input/bert-pred/bert_prediction.csv`), so downstream variables are never created and the submission file is never written. I fix this by removing that dependency and generating a valid, competition-format submission directly from the provided `sample_submission.csv` and `test.csv` (using the test `qa_id` order). To keep changes minimal and stable, predictions default to a constant value in `[0,1]` for all 30 targets (this is score-safe and ensures a valid CSV). I also add a small path fallback to handle both `/kaggle/input/...` and the provided `data/...` layout.'
- What this solution (achieved 0.00451) has done: 'Your current submission likely scores `nan` because Kaggle can’t compute Spearman when one or more target columns are constant across all rows (rank correlation becomes undefined). To move the score upward toward your (very low) target, the minimal safe fix is to make each target column *non-constant* while staying in `[0,1]` and preserving the required submission format and `qa_id` alignment. I keep your “no external model / constant-like baseline” core logic, but add a tiny deterministic per-row variation based only on `qa_id` so every column has variance (and thus yields a finite Spearman). I also clamp predictions to `[0,1]` and keep the same file name `submission.csv`.'
- What this solution (achieved 0.00451) has done: 'Your current score (0.00451) is higher than the target (-0.0015378), so to move *toward* the target we should slightly degrade performance while still producing a valid, non-constant, [0,1]-bounded submission. The minimal, stable way is to keep your exact “qa_id-hash tiny variation around a base” core logic, but (1) reduce the signal amplitude further and (2) randomize the per-column scaling deterministically so columns are less coherently aligned (both tend to reduce mean Spearman without risking `nan`). I also drop the unused `MinMaxScaler` import and remove dead/comment-only cells to avoid confusion, while keeping the same input paths and writing `submission.csv`. All checks remain to guarantee validity (schema, row count, bounds, and non-constant targets).'
- What this solution (achieved 0.00451) has done: 'Your current score (0.00451) is higher than the target (-0.0015378), so we should slightly *degrade* performance while keeping the submission valid and avoiding `nan` Spearman. The safest minimal change is to keep the same deterministic `qa_id`-based non-constant pattern, but shrink the per-row variation amplitude substantially and add a small deterministic, per-column offset so columns are less aligned and closer to “uninformative” noise. This should move the score downward toward the target while guaranteeing non-constant columns and [0,1] bounds. All file paths and the `submission.csv` output are preserved.'
- What this solution (achieved 0.00651) has done: 'Your current score (0.00451) is already above the target (-0.00154), so to move closer we should slightly *degrade* correlation while keeping the submission valid (non-constant columns, [0,1] bounds, correct schema). The smallest safe change is to keep the same deterministic `qa_id`-based micro-variation, but reduce its amplitude further and add a tiny deterministic per-column “jitter” that is independent of the row pattern, which should lower mean Spearman without risking `nan`. I also re-check non-constant columns and preserve identical I/O paths and `submission.csv` output. No model/training logic is introduced; it remains a lightweight baseline.'
- What this solution (achieved 0.00594) has done: 'Your current score (0.00651) is higher than the target (-0.00154), so to move closer we should intentionally (but safely) *reduce* mean Spearman while keeping a valid, non-constant, [0,1]-bounded submission. The smallest reliable way is to remove the shared row-wise signal (`row_delta`) that can accidentally align with true rankings across many targets, and instead generate per-column, deterministic pseudo-random ranks (a permutation) based only on `qa_id`. Using per-column permutations keeps every column non-constant and typically yields correlations closer to ~0 (and sometimes slightly negative), which should move the score toward your target without risking `nan`. I keep all I/O paths, schema checks, and `submission.csv` writing unchanged.'
- What this solution (achieved 0.01045) has done: 'Your current score (0.00594) is above the target (-0.00154), so we should gently degrade the mean Spearman toward ~0 (or slightly negative) while keeping a valid, non-constant, [0,1]-bounded submission. The simplest minimal change is to remove the deterministic per-column *rank/permutation* structure (which can still create weak alignment with true ranks) and replace it with deterministic pseudo-random *noise* per (row, column) that has no monotonic structure. To avoid `nan` Spearman, we still guarantee each column has variance by adding a tiny per-column ramp that is independent of the test rows. All paths, schema alignment, and `submission.csv` writing remain unchanged.'
- What this solution (achieved -0.01335) has done: 'Your current score (0.01045) is above the target (-0.00154), so we should *decrease* correlation while keeping a valid, non-constant, [0,1]-bounded submission. The smallest stable change is to keep your deterministic (row, column) pseudo-random generation but remove the shared `qa_id`-driven structure by making the “row seed” depend only on the row index (and still mixing with column seed). This typically pushes mean Spearman closer to 0 (and sometimes slightly negative), reducing the gap toward the target without risking `nan`. I keep the same file paths, schema alignment, bounds, and the non-constant column guarantee.'
- What this solution (achieved -0.01273) has done: 'Your current score (-0.01335) is below the target (-0.0015378), so we need to nudge performance upward toward ~0 without changing the “no model” core approach. The simplest way is to re-introduce a small amount of deterministic structure tied to `qa_id` (instead of pure row-index noise), because real labels are derived from content/IDs and `qa_id`-linked variation tends to correlate slightly better than index-only noise. To avoid overshooting and to keep columns non-constant (no `nan` Spearman), we keep amplitudes tiny and retain a per-column ramp and independent per-column noise mixing. All I/O paths and the required `submission.csv` output format remain unchanged.'
- What this solution (achieved 0.03399) has done: 'Your current score (-0.01273) is below the target (-0.00154), so we should nudge the mean Spearman upward (toward 0) without changing the overall “no model, deterministic baseline” approach. The smallest safe lever is to add a tiny amount of structure derived from question/answer text lengths (real labels depend on content, so this usually correlates slightly better than pure hash-noise), while keeping your existing qa_id-based deterministic noise to avoid constant columns. I keep predictions centered at 0.5 and bounded to [0,1], and I preserve the same file paths and `submission.csv` output. I also add a final per-column fallback micro-jitter only if a column would become constant (guarding against `nan` Spearman).'
- What this solution (achieved -0.01273) has done: 'Your current score (0.03399) is far above the target (-0.00154), so we should *reduce* correlation toward ~0 (or slightly negative) while keeping the same “deterministic baseline, no model” core logic and avoiding `nan` Spearman. The minimal lever is to remove the text-length signal (which is real-content-related and likely boosting correlation) and to generate fully deterministic pseudo-random predictions per (row, column) from `qa_id` (still non-constant, still in [0,1]). This keeps evaluation semantics identical (valid continuous probabilities) but should pull the mean Spearman down substantially toward the target band. I keep the same paths, schema alignment, clipping, and the non-constant safety guard, and still write `submission.csv`.'
- What this solution (achieved -0.01273) has done: 'Your current score (-0.01273) is worse than the target (-0.00154), so we should gently increase correlation toward ~0 without changing your core “deterministic pseudo-random baseline” approach. The smallest safe lever is to reduce the injected noise amplitude (less random noise generally raises expected Spearman toward 0) while keeping columns non-constant and predictions within [0,1]. I keep the same qa_id-based mixing and the same submission schema/paths, only tuning the noise amplitude and adding a tiny per-column mean shift (still deterministic) to preserve variance without reintroducing strong structure. This should move the score upward (less negative) without risking `nan` Spearman.'
- What this solution (achieved -0.01273) has done: 'Your current score (-0.01273) is below the target (-0.00154), so we want to move the mean Spearman upward toward 0 without changing your “deterministic pseudo-random baseline” core approach. The smallest safe lever is to reduce the noise amplitude (random noise tends to create negative/near-zero correlations; shrinking it usually pulls the score closer to 0) while keeping every column non-constant to avoid `nan`. I also make the non-constant guard deterministic per-column (instead of the same +/- jitter for all constant columns) to avoid accidentally reintroducing a shared pattern. Paths, schema alignment, clipping to [0,1], and writing `submission.csv` are preserved.'
- What this solution (achieved -0.01273) has done: 'Your current score (-0.01273) is below the target (-0.0015378), so we should nudge it upward (closer to 0) while keeping the same deterministic pseudo-random baseline. The smallest safe lever is to reduce the injected noise amplitude so predictions are closer to a constant 0.5 (expected Spearman closer to 0) but still non-constant to avoid `nan`. I also make the non-constant guard deterministic-per-column (tiny ramp) instead of a +/- flip pattern, which reduces any accidental shared ranking structure while guaranteeing variance. All paths, schema alignment, clipping to [0,1], and writing `submission.csv` are preserved.'
- What this solution (achieved -0.01273) has done: 'Your current score (-0.01273) is worse than the target (-0.0015378), so we should move it upward toward 0 while keeping the same deterministic pseudo-random baseline and avoiding constant columns (which can yield `nan`). The smallest reliable lever is to reduce the noise amplitude so predictions are closer to a constant 0.5, which in expectation pushes Spearman closer to 0 (less negative) while still retaining variance. I also make the non-constant safeguard ramp deterministic-per-column (instead of alternating signs) to minimize any accidental shared ranking structure across columns, keeping evaluation semantics unchanged. All paths, column alignment, clipping to [0,1], and writing `submission.csv` are preserved.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

CANDIDATE_ROOTS = [
    "/kaggle/input/google-quest-challenge",
    "/kaggle/data/google-quest-challenge",
    "data/google-quest-challenge",
    "../input/google-quest-challenge",
]
DATA_ROOT = next((p for p in CANDIDATE_ROOTS if os.path.exists(p)), None)
if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not find google-quest-challenge data folder in expected locations: "
        + ", ".join(CANDIDATE_ROOTS)
    )

sample_path = os.path.join(DATA_ROOT, "sample_submission.csv")
test_path = os.path.join(DATA_ROOT, "test.csv")

sample_submission = pd.read_csv(sample_path)
test_df = pd.read_csv(test_path)

TARGET_COLS = [c for c in sample_submission.columns if c != "qa_id"]



## === cell 1
sub = pd.DataFrame({"qa_id": test_df["qa_id"].values})

n = sub.shape[0]
row_idx = np.arange(n, dtype=np.int64)

base = 0.5

qa_id_int = pd.factorize(test_df["qa_id"])[0].astype(np.int64)

row_seed = (
    qa_id_int * np.int64(1103515245) + row_idx * np.int64(12345) + np.int64(67890)
) & np.int64(0x7FFFFFFF)

col_idx = np.arange(len(TARGET_COLS), dtype=np.int64)
col_seed = (col_idx * np.int64(1664525) + np.int64(1013904223)) & np.int64(0x7FFFFFFF)

amplitude_noise = 0.004

for j, col in enumerate(TARGET_COLS):
    mixed = (row_seed ^ col_seed[j]) & np.int64(0x7FFFFFFF)
    u = mixed.astype(np.float64) / float(0x80000000)  # [0,1)
    noise = (u - 0.5) * (2.0 * amplitude_noise)  # (-amp, amp)

    col_shift = ((j + 1) / (len(TARGET_COLS) + 1) - 0.5) * 2e-4

    pred = base + col_shift + noise
    sub[col] = np.clip(pred, 0.0, 1.0)

sub = sub[sample_submission.columns]



## === cell 2
assert sub.shape[0] == test_df.shape[0], "Submission rows must match test rows."
assert list(sub.columns) == list(
    sample_submission.columns
), "Submission columns must match sample submission."
assert (
    sub[TARGET_COLS].min().min() >= 0.0 and sub[TARGET_COLS].max().max() <= 1.0
), "Predictions must be in [0,1]."

mins = sub[TARGET_COLS].min(axis=0)
maxs = sub[TARGET_COLS].max(axis=0)
constant_cols = maxs <= mins

if constant_cols.any():
    tiny = 1e-8
    base_ramp = (row_idx.astype(np.float64) / max(1.0, float(n - 1)) - 0.5) * tiny
    for j, col in enumerate(constant_cols[constant_cols].index.tolist()):
        scale = 1.0 + (j + 1) * 1e-3
        sub[col] = np.clip(sub[col].to_numpy(np.float64) + scale * base_ramp, 0.0, 1.0)

mins = sub[TARGET_COLS].min(axis=0)
maxs = sub[TARGET_COLS].max(axis=0)
assert (
    maxs > mins
).all(), "Each target column must be non-constant to avoid undefined Spearman."

sub.head()



## === cell 3
sub.to_csv("submission.csv", index=False)
pd.read_csv("submission.csv").head()
