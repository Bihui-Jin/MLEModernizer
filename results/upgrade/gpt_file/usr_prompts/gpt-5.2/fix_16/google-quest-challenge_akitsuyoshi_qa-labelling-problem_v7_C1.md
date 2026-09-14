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

3.11

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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
seaborn==0.12.2
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

0.15083

# 6. Current score

-0.00802

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.29735) has done: 'Your current score (0.28781) is substantially higher than the target (0.15083), so the goal is to *reduce* performance in a controlled, minimal way while keeping the same overall pipeline (TF-IDF → SVD → MultiOutput KNN). The smallest reliable lever is to make the KNN model more biased/smoother by increasing `n_neighbors`, which typically lowers rank-correlation without changing the approach. I also add a fixed `random_state` to the train/validation split to keep the measured effect stable run-to-run, and I clip predictions to `[0,1]` to match submission requirements (usually negligible but safer). The script still run end-to-end and write `submission.csv` with the correct columns.'
- What this solution (achieved 0.3022) has done: 'Your current public score (0.29735) is well above the target (0.15083), so to move *toward* the target we should gently reduce model performance without changing the pipeline (TF‑IDF → SVD → MultiOutput KNN). The smallest stable lever is to further increase `n_neighbors`, which makes KNN predictions smoother/more biased and typically lowers Spearman rank correlation. I’m keeping everything else the same (same features, same SVD size, same MultiOutputRegressor), and I’m keeping clipping to `[0,1]` and the existing `random_state` to keep behavior consistent. The script still runs end-to-end and writes a valid `submission.csv` with the correct columns.'
- What this solution (achieved 0.28295) has done: 'Your current score (0.3022) is well above the target (0.15083), so we should *reduce* performance in a controlled, minimal way while keeping the exact same pipeline (TF‑IDF → SVD → MultiOutput KNN). The smallest, most reliable lever is to further increase `n_neighbors`, which makes KNN predictions smoother and typically lowers the mean Spearman correlation without changing the modeling approach. I keep everything else identical (same features, vectorizer, SVD size, training flow), and keep clipping to `[0,1]` plus the fixed split seed for stability. This still runs end-to-end and writes a valid `submission.csv` with the correct columns.'
- What this solution (achieved nan) has done: 'I fix the runtime error by ensuring `n_neighbors` never exceeds the number of training samples (both in the train/validation split and in the full-train fit), while keeping the exact same TF‑IDF → SVD → MultiOutput KNN pipeline. Because your target score (0.15083) is lower than the previously observed scores for this approach, I keep the intentionally “smoother/worse” behavior by using a large neighbor count, but cap it safely to `n_samples_fit` to avoid crashes. I also make the text concatenation robust to missing values by filling NA strings, and keep clipping predictions to `[0,1]` to satisfy submission requirements. Finally, the script always write a valid `submission.csv` with the correct header/columns.'
- What this solution (achieved nan) has done: 'Your current run can yield `nan` because Spearman correlation returns `nan` when a prediction column is constant (which can happen with very large KNN neighbor counts that average everything). To keep the same TF‑IDF → SVD → MultiOutput KNN core logic while moving the score toward the target, I (1) make the validation Spearman computation robust by mapping any `nan` correlations to `0.0` (so you can monitor progress), and (2) slightly reduce the smoothing by lowering `desired_k` from 12000 to a still-large value (8000), which should increase variance in predictions and avoid constant columns while still keeping performance relatively low. The submission pipeline remains identical and still writes a valid `submission.csv` with the correct columns and `[0,1]` clipping.'
- What this solution (achieved 0.28432) has done: 'To move your score upward from `nan` toward the target (0.15083), the key is to prevent constant prediction columns, which make Spearman correlation undefined (`nan`) and can also lead to `nan` public scores. Keeping the exact same TF‑IDF → SVD → MultiOutput KNN pipeline, the smallest effective change is to slightly reduce over-smoothing by lowering `desired_k` so predictions keep enough variance. I also add a tiny deterministic jitter to predictions *only when a column is (near-)constant* to guarantee Spearman is defined, without changing the model/training approach. The submission writing path/format stays the same and still clips predictions to `[0,1]`.'
- What this solution (achieved -0.00802) has done: 'Your current score (0.28432) is well above the target (0.15083), so we should *reduce* performance in a controlled, minimal way while keeping the exact TF‑IDF → SVD → MultiOutput KNN pipeline unchanged. The smallest reliable lever is to increase `desired_k` (neighbors), which smooths predictions and typically lowers mean Spearman correlation. To avoid the earlier `nan`/undefined Spearman issues from constant columns, I keep the existing “near-constant column jitter” safeguard, and I also cap `k` safely to `n_samples_fit` in both split and full training. Everything else (features, vectorizer, SVD size, model class, train flow, submission format) remains the same and still writes `submission.csv`.'
- What this solution (achieved 0.30275) has done: 'We need to move your public score up from -0.00802 toward 0.15083 (higher-is-better) while keeping the same TF‑IDF → SVD → MultiOutput KNN core. The most likely reason for the negative score is that `desired_k=8000` is so large that predictions become overly smoothed/near-constant, destroying rank ordering (Spearman) even if values stay in [0,1]. I make the smallest change that reliably increases rank-variance: reduce `desired_k` to a still-large but much less smoothing value (200) and slightly relax the “near-constant” detection threshold so the jitter safeguard triggers earlier when needed. Everything else (features, vectorizer, SVD size, model class, training flow, submission format/path) remains unchanged and it still writes `submission.csv`.'
- What this solution (achieved 0.2989) has done: 'Your current score (0.30275) is far above the target (0.15083), so we should intentionally and gently *reduce* Spearman performance while keeping the exact same TF‑IDF → SVD → MultiOutput KNN pipeline. The smallest reliable lever is to increase `n_neighbors` so predictions become smoother and rankings degrade in a controlled way; I’m setting `desired_k` to a larger value while still capping it safely to avoid crashes. Because heavy smoothing can create (near-)constant columns that break Spearman, I keep your jitter safeguard but make it trigger earlier (slightly higher threshold) so the submission stays valid and avoids pathological constant outputs. Everything else (features, vectorizer, SVD size, model class, training flow, output format, and `[0,1]` clipping) is unchanged.'
- What this solution (achieved 0.26661) has done: 'Your current score (0.2989) is far above the target (0.15083), so we should intentionally reduce performance in a controlled, minimal way while keeping the exact TF‑IDF → SVD → MultiOutput KNN pipeline unchanged. The smallest reliable lever is to increase `n_neighbors`, which smooths predictions more and typically lowers mean Spearman correlation without changing model class or training procedure. To avoid the earlier `nan` issues from (near-)constant prediction columns, we keep the same jitter safeguard and clipping to `[0,1]`, and we keep deterministic `random_state` for stability. The code still runs end-to-end and writes a valid `submission.csv` with the correct columns.'
- What this solution (achieved -0.00802) has done: 'Your current score (0.26661) is well above the target (0.15083), so we should intentionally *reduce* performance in a controlled, minimal way while keeping the exact same TF‑IDF → SVD → MultiOutput KNN pipeline. The smallest reliable lever is to further increase `n_neighbors`, which makes predictions smoother and typically lowers mean Spearman correlation without changing the modeling approach. To avoid the earlier `nan`/undefined Spearman issues from constant columns, I keep your near-constant-column jitter safeguard (slightly more conservative threshold) and keep clipping to `[0,1]`. Everything else (feature construction, vectorizer/SVD, train/test split, model class, output format/path) remains unchanged and it still write a valid `submission.csv`.'
- What this solution (achieved 0.30119) has done: 'We need to move your score up from -0.00802 toward 0.15083 (higher-is-better), and the most likely cause of the negative Spearman is over-smoothing from an extremely large `n_neighbors=12000`, which collapses rankings. I make the smallest change that increases rank-variance while keeping the exact same TF‑IDF → SVD → MultiOutput KNN pipeline: reduce `desired_k` to a moderate value (500) and keep the same caps. I also make the “near-constant column jitter” slightly stronger (still tiny and deterministic) so we avoid degenerate constant columns that can tank Spearman, while still clipping to [0,1] for valid submissions. Everything else (features, vectorizer/SVD, training flow, file paths, output schema) stays the same and it still write `submission.csv`.'
- What this solution (achieved 0.27834) has done: 'Your current score (0.30119) is far above the target (0.15083), so we should intentionally reduce performance in a controlled, minimal way while keeping the exact same TF‑IDF → SVD → MultiOutput KNN pipeline. The smallest reliable lever is to increase `n_neighbors`, which smooth predictions and degrade rank ordering (Spearman) without changing the approach. To avoid previous `nan`/degenerate behavior, we keep the same neighbor cap and the same near-constant-column jitter+clipping safeguards. Everything else (feature construction, vectorizer/SVD settings, training flow, submission schema/path) remains unchanged and it still write a valid `submission.csv`.'
- What this solution (achieved -0.00802) has done: 'Your current score (0.27834) is well above the target (0.15083), so we should intentionally lower performance in a controlled, minimal way while keeping the same TF‑IDF → SVD → MultiOutput KNN pipeline. The smallest reliable lever is to increase `n_neighbors` to make predictions smoother and degrade rank ordering (Spearman) without changing the modeling approach. To prevent crashes and keep behavior stable, I keep the existing caps (`min(desired_k, n_samples_fit)`), fixed `random_state`, and the near-constant-column jitter+clipping safeguards. Everything else (features, vectorizer/SVD settings, training flow, and submission schema/path) remains unchanged and still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re, string
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import spearmanr

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train = pd.read_csv("/kaggle/input/google-quest-challenge/train.csv", low_memory=True)
test = pd.read_csv("/kaggle/input/google-quest-challenge/test.csv", low_memory=True)
subm = pd.read_csv(
    "/kaggle/input/google-quest-challenge/sample_submission.csv", low_memory=True
)
train.shape, test.shape, subm.shape



## === cell 2
target_cols = subm.columns[1:]
feat_cols = test.columns[1:]
target_cols, feat_cols



## === cell 3
train.head(4)



## === cell 4
print(train["question_title"][1])
print("------")
print(train["question_body"][1])
print("------")
print(train["answer"][1])



## === cell 5
print(train["question_title"][100])
print("-----")
print(train["question_body"][100])
print("-----")
print(train["answer"][100])



## === cell 6
lens_quest = train.question_body.str.len()
sns.histplot(lens_quest)
plt.title("Question Body Length")
plt.show()
lens_quest.min(), lens_quest.mean(), lens_quest.max(), lens_quest.std()



## === cell 7
lens_ans = train.answer.str.len()
sns.histplot(lens_ans)
plt.title("Answer Body Length")
plt.show()
lens_ans.min(), lens_ans.mean(), lens_ans.max(), lens_ans.std()



## === cell 8
train[train.question_title.str.len() < 10]



## === cell 9
train[train.question_body.str.len() < 10]



## === cell 10
train[train.answer.str.len() < 30].answer



## === cell 11
train.describe()



## === cell 12
train.info()



## === cell 13
train[train.isna().sum(axis=1) == 1]



## === cell 14
colors = sns.color_palette("pastel")[0:5]
plt.pie(
    train.category.value_counts(),
    labels=train.category.value_counts().index,
    colors=colors,
    autopct="%.0f%%",
)
plt.show()



## === cell 15
train["host"] = train["host"].fillna("")
test["host"] = test["host"].fillna("")
train["host_type"] = train.host.apply(
    lambda x: x.split(".")[0] if isinstance(x, str) and len(x) else ""
)
test["host_type"] = test.host.apply(
    lambda x: x.split(".")[0] if isinstance(x, str) and len(x) else ""
)

print("Top Genre: ")
print(train.host_type.value_counts()[:5])
print("Lower Genre: ")
print(train.host_type.value_counts()[-5:])



## === cell 16
sns.kdeplot(train.answer_well_written)
sns.kdeplot(train.question_well_written)
plt.legend()
plt.show()



## === cell 17
re_tok = re.compile(f"([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])")


def tokenize(s):
    return re_tok.sub(r" \1 ", s).split()


vect = TfidfVectorizer(
    ngram_range=(1, 2),
    tokenizer=tokenize,
    min_df=round(len(train) * 0.001),
    max_df=0.9,
    sublinear_tf=True,
)

tsvd = TruncatedSVD(100)



## === cell 18
text_cols = [
    "host_type",
    "question_title",
    "question_body",
    "category",
    "question_user_name",
    "answer",
    "answer_user_name",
]
for c in text_cols:
    train[c] = train[c].fillna("").astype(str)
    test[c] = test[c].fillna("").astype(str)

train["QA"] = (
    "xxhostxx"
    + train["host_type"]
    + "xxqtixx "
    + train["question_title"]
    + "xxqbodxx "
    + train["question_body"]
    + "xxcategoryxx "
    + train["category"]
    + "xxq_userxx "
    + train["question_user_name"]
    + "xxansxx "
    + train["answer"]
    + "xxans_userxx "
    + train["answer_user_name"]
)

test["QA"] = (
    "xxhostxx"
    + test["host_type"]
    + "xxqtixx "
    + test["question_title"]
    + "xxqbodxx "
    + test["question_body"]
    + "xxcategoryxx "
    + test["category"]
    + "xxq_userxx "
    + test["question_user_name"]
    + "xxansxx "
    + test["answer"]
    + "xxans_userxx "
    + test["answer_user_name"]
)

vected_train = vect.fit_transform(train["QA"])
vected_test = vect.transform(test["QA"])

vected_train = tsvd.fit_transform(vected_train)
vected_test = tsvd.transform(vected_test)



## === cell 19
X_train, X_test, y_train, y_test = train_test_split(
    vected_train, train[target_cols], test_size=0.3, random_state=42
)
X_train.shape, X_test.shape




## === cell 20
def spearmancoff(y_pred, y_true):
    vals = []
    for i in range(y_true.shape[1]):
        rho = spearmanr(y_pred[:, i], y_true.iloc[:, i]).correlation
        if rho is None or np.isnan(rho):
            rho = 0.0
        vals.append(rho)
    return float(np.mean(vals))


desired_k = 9000

k_split = min(desired_k, X_train.shape[0])

estimator = KNeighborsRegressor(
    n_neighbors=k_split,
    weights="uniform",
    metric="minkowski",
)
model = MultiOutputRegressor(estimator)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
spearmancoff(y_pred, y_test)



## === cell 21
k_full = min(desired_k, vected_train.shape[0])

estimator = KNeighborsRegressor(
    n_neighbors=k_full,
    weights="uniform",
    metric="minkowski",
)
model = MultiOutputRegressor(estimator)
model.fit(vected_train, train[target_cols])

y_pred = model.predict(vected_test)

col_std = y_pred.std(axis=0)
near_const = col_std < 1e-4
if np.any(near_const):
    rng = np.random.RandomState(42)
    jitter = rng.normal(loc=0.0, scale=1e-4, size=y_pred[:, near_const].shape)
    y_pred[:, near_const] = y_pred[:, near_const] + jitter

y_pred = np.clip(y_pred, 0.0, 1.0)

submission = pd.concat(
    [
        pd.DataFrame({"qa_id": test["qa_id"]}),
        pd.DataFrame(y_pred, columns=target_cols),
    ],
    axis=1,
)

submission = submission[["qa_id"] + list(target_cols)]
submission.to_csv("submission.csv", index=False)

submission.head()



## === cell 22
submission
