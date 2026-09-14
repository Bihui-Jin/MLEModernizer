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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scipy==1.15.3
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

0.00022

# 6. Current score

0.00056

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.00203) has done: 'I fix the undefined variables error, correctly compute Spearman correlations (using only the coefficient), and generate test‑set predictions by sampling from each target column’s empirical distribution in the training data instead of pure random numbers. This retains the original simple baseline logic while giving a small positive correlation boost, moving the score from -0.00731 toward the target 0.00022. The script now produces a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.18408) has done: 'I replace the random‑sampling prediction step with a simple categorical‑mean baseline: for each target column I compute the mean value per `category` in the training set and use that as the prediction for test rows belonging to the same category (falling back to the global column mean for unseen categories). This deterministic, slightly more informed approach should raise the Spearman correlation toward the target positive value while preserving the overall pipeline and output format.'
- What this solution (achieved 0.03461) has done: 'I reduce the correlation by blending the deterministic category‑mean predictions with random noise. This keeps the overall pipeline intact but moves the Spearman score down toward the target near 0.00022. A fixed random seed ensures reproducible output.'
- What this solution (achieved -0.01076) has done: 'I lower the blend factor used when combining the deterministic category‑mean predictions with random noise, turning the model into almost‑pure random predictions. This reduces the Spearman correlation from the current positive value toward zero, bringing the score much closer to the target 0.00022 while keeping the overall pipeline unchanged and still writing a valid `submission.csv`. I also keep the random seed for reproducibility.'
- What this solution (achieved -0.01022) has done: 'I add a small validation split to tune the blend factor between deterministic category‑mean predictions and random noise, selecting the factor that yields a mean Spearman correlation closest to the target (0.00022). This keeps the original pipeline but nudges the score upward from the current negative value without overshooting far beyond the tiny target.'
- What this solution (achieved -0.00367) has done: 'I expand the blend‑factor search to a finer grid (0 → 0.30 step 0.02) and make the evaluation deterministic by resetting the random seed inside the evaluation function. This lets the script pick a blend factor that moves the validation Spearman correlation nearer to the target 0.00022, thereby improving the final score while preserving the original pipeline.'
- What this solution (achieved -0.00367) has done: 'I broaden the blend‑factor grid to search more finely (0 → 0.50 in steps of 0.01) so the validation‐set Spearman score can be tuned closer to the tiny target 0.00022, while keeping the rest of the pipeline unchanged. This small change preserves the core logic and should move the final score toward the target without overshooting.'
- What this solution (achieved 0.00056) has done: 'I replace the uniform random noise with per‑column sampling from the training label distributions, then keep the existing blend‑factor search (now using more informative noise). This small change should raise the validation Spearman correlation toward the tiny positive target without altering the overall pipeline.'
- What this solution (achieved 0.00056) has done: 'I slightly lower the blend factor when the validation correlation is above the target, so the final predictions become a bit noisier and the Spearman score moves closer to the target (reducing the current gap). This keeps the original logic intact while nudging performance in the desired direction.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE_PATH = "../input/google-quest-challenge/"
TRAIN_PATH = BASE_PATH + "train.csv"
TEST_PATH = BASE_PATH + "test.csv"
SUB_PATH = BASE_PATH + "sample_submission.csv"

TARGET_SCORE = 0.00022




## === cell 1
print("\n# Files and file sizes")
for file in os.listdir(BASE_PATH):
    print(
        f"{file.ljust(30)}| {round(os.path.getsize(BASE_PATH + file) / 1_000_000, 2)} MB"
    )




## === cell 2
target_cols = [
    "question_asker_intent_understanding",
    "question_body_critical",
    "question_conversational",
    "question_expect_short_answer",
    "question_fact_seeking",
    "question_has_commonly_accepted_answer",
    "question_interestingness_others",
    "question_interestingness_self",
    "question_multi_intent",
    "question_not_really_a_question",
    "question_opinion_seeking",
    "question_type_choice",
    "question_type_compare",
    "question_type_consequence",
    "question_type_definition",
    "question_type_entity",
    "question_type_instructions",
    "question_type_procedure",
    "question_type_reason_explanation",
    "question_type_spelling",
    "question_well_written",
    "answer_helpful",
    "answer_level_of_information",
    "answer_plausible",
    "answer_relevance",
    "answer_satisfaction",
    "answer_type_instructions",
    "answer_type_procedure",
    "answer_type_reason_explanation",
    "answer_well_written",
]




## === cell 3
df = pd.read_csv(TRAIN_PATH)




## === cell 4
print("Target variables head:")
print(df[target_cols].head())




## === cell 5
def spearmans_rho(y_true, y_pred):
    """Return only the Spearman correlation coefficient."""
    return spearmanr(y_true, y_pred).correlation




## === cell 6
def _get_ranks(arr: np.ndarray) -> np.ndarray:
    temp = arr.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(arr))
    return ranks


def spearmans_rho_custom(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    true_rank = _get_ranks(y_true)
    pred_rank = _get_ranks(y_pred)
    return np.corrcoef(true_rank, pred_rank)[1, 0]




## === cell 7
corrs = []
for col in target_cols:
    probs = df[col].value_counts(normalize=True).values
    vals = df[col].value_counts().index.to_numpy()
    naive_preds = np.random.choice(vals, size=len(df), p=probs)
    corr = spearmans_rho(df[col], naive_preds)
    corrs.append(corr)
avg_corr = np.mean(corrs)
print(f"Baseline average Spearman correlation (training‑set sampling): {avg_corr:.6f}")




## === cell 8
test_df = pd.read_csv(TEST_PATH)
sub_df = pd.DataFrame()
sub_df["qa_id"] = test_df["qa_id"]




## === cell 9
np.random.seed(42)

val_mask = np.random.rand(len(df)) < 0.2
val_df = df[val_mask].reset_index(drop=True)
train_df = df[~val_mask].reset_index(drop=True)

category_means = train_df.groupby("category")[target_cols].mean()
global_means = train_df[target_cols].mean()

empirical_vals = {}
empirical_probs = {}
for col in target_cols:
    vc = train_df[col].value_counts(normalize=True)
    empirical_vals[col] = vc.index.to_numpy()
    empirical_probs[col] = vc.values


def evaluate_blend(blend):
    """Return mean Spearman correlation on the validation set for a given blend."""
    np.random.seed(42)
    corrs = []
    for col in target_cols:
        preds = val_df["category"].map(category_means[col]).fillna(global_means[col])
        noise = np.random.choice(
            empirical_vals[col], size=len(preds), p=empirical_probs[col]
        )
        preds = blend * preds + (1 - blend) * noise
        preds = np.clip(preds, 0.0, 1.0)
        corr = spearmans_rho(val_df[col], preds)
        corrs.append(corr)
    return np.mean(corrs)


candidate_factors = np.arange(0.0, 0.51, 0.01)
best_factor = 0.0
best_gap = float("inf")
best_corr = None
for bf in candidate_factors:
    avg_corr_val = evaluate_blend(bf)
    gap = abs(avg_corr_val - TARGET_SCORE)
    print(
        f"Blend {bf:.2f}: validation mean Spearman = {avg_corr_val:.6f}, gap = {gap:.6f}"
    )
    if gap < best_gap:
        best_gap = gap
        best_factor = bf
        best_corr = avg_corr_val

print(f"Selected blend_factor = {best_factor:.2f} (closest to target {TARGET_SCORE})")

if best_corr is not None and best_corr > TARGET_SCORE:
    adjusted_factor = max(0.0, best_factor - 0.01)  # small step downwards
    print(
        f"Validation correlation {best_corr:.6f} > target {TARGET_SCORE:.6f}; "
        f"adjusting blend_factor from {best_factor:.2f} to {adjusted_factor:.2f}"
    )
    blend_factor = adjusted_factor
else:
    blend_factor = best_factor

np.random.seed(42)  # reproducibility for final predictions
for col in target_cols:
    preds = test_df["category"].map(category_means[col]).fillna(global_means[col])
    noise = np.random.choice(
        empirical_vals[col], size=len(preds), p=empirical_probs[col]
    )
    preds = blend_factor * preds + (1 - blend_factor) * noise
    preds = np.clip(preds, 0.0, 1.0)
    sub_df[col] = preds




## === cell 10
sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print(f"Submission file written to {sub_path}")




## === cell 11
print("First few rows of the submission:")
print(sub_df.head(2))
