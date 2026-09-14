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
scipy==1.15.3
sklearn-pandas==2.2.0
tqdm==4.67.1

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

0.29392

# 6. Current score

0.35274

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.39013) has done: 'I fixed the incompatibility with the current SciPy version by switching the Ridge regression solver to **'lsqr'**, which works with sparse matrices without calling the problematic `cg` function. I also added a safety clip to keep all predictions inside the required `[0, 1]` range. These changes allow the training loops to run completely, producing all required columns in the submission file and eliminating the subsequent KeyError.'
- What this solution (achieved 0.38872) has done: 'I slightly increase the regularisation strength for every target by scaling the Ridge α values by 1.5. This modest change should reduce over‑fitting, lowering the validation Spearman/AUC scores and moving the overall metric closer to the target (while keeping the core pipeline intact). I also add a tiny comment explaining the purpose.'
- What this solution (achieved 0.38471) has done: 'I increase the regularisation strength used for every target by changing the scaling factor from 1.5 to 2.5 when the scaled_alphas dictionary is built. A larger alpha makes the Ridge models under‑fit a bit, which reduces the Spearman correlation and AUC scores, moving the current metric (0.38872) closer to the target range (≈0.2645‑0.3233) without altering any core logic or feature handling.'
- What this solution (achieved 0.38155) has done: 'I slightly increase the regularisation strength for every target by scaling the original α values with a larger factor (3.5 instead of 2.5). This under‑fits the Ridge models a bit more, which lowers the Spearman correlation and therefore moves the score closer to the target (without altering the core pipeline, feature handling, or prediction format).'
- What this solution (achieved 0.3683) has done: 'I increase the regularisation strength for every target by raising the scaling factor applied to the original alphas from 3.5 to 12.0. This stronger α cause the Ridge models to under‑fit further, lowering the Spearman/AUC scores and moving the overall metric from the current 0.38155 closer to the target 0.29392 while preserving the core pipeline unchanged. The only modification is in cell 8 where the scaled alphas dictionary is created.'
- What this solution (achieved 0.36053) has done: 'I increase the regularisation scaling factor used to build `scaled_alphas` from 12.0 to 30.0. A larger α makes each Ridge model under‑fit more, which reduces the Spearman correlation and moves the evaluation score from the current 0.3683 down toward the target 0.2939 while keeping the original pipeline intact.'
- What this solution (achieved 0.35821) has done: 'The score is higher than the target, so we need to reduce it.  
Increasing the regular‑isation strength a bit more (scaling factor ≈ 45 instead of 30) makes each Ridge model under‑fit slightly more, which lowers the validation Spearman/AUC and moves the overall metric into the target band while keeping the pipeline unchanged.'
- What this solution (achieved 0.35595) has done: 'I increase the regularisation strength further by raising the scaling factor used for the Ridge α values from 45 to 80. This slightly stronger α under‑fit the models a bit more, lowering the validation Spearman/AUC scores and moving the overall metric from 0.358 toward the target range (~0.30) while keeping the core pipeline unchanged.'
- What this solution (achieved 0.35485) has done: 'The scaling factor for the Ridge regularisation is increased to make each model under‑fit more, which reduces the validation Spearman/AUC scores and moves the overall metric closer to the target (lowering it from 0.35595 toward 0.29392) while keeping the core pipeline unchanged.'
- What this solution (achieved 0.3539) has done: 'I increase the regularisation scaling factor used for the Ridge α values (cell 8) from 120 to 200. A larger α makes each model under‑fit more, which lowers the validation AUC and Spearman scores, moving the overall metric from 0.35485 closer to the target 0.29392 while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.35341) has done: 'I keep the whole pipeline unchanged and only increase the regularisation strength a bit more so the model under‑fits further, which lowers the Spearman/AUC scores and moves the current 0.3539 closer to the target 0.2939. The only modification is in the cell that builds `scaled_alphas`: the scaling factor is raised from 200 to 300. This small change preserves all core logic while reducing the overall score toward the desired range.'
- What this solution (achieved 0.35299) has done: 'I increase the regularisation scaling factor used for the Ridge α values so that each model under‑fits a bit more, which lowers the validation AUC and Spearman scores and moves the overall metric closer to the target (since higher is better). The only change is in **cell 8**, where the factor is raised from 300 to 500. This keeps the core pipeline, feature engineering, and prediction logic unchanged while adjusting model strength to reduce the score toward the desired 0.29392.'
- What this solution (achieved 0.35274) has done: 'I keep the whole pipeline unchanged and only make the regularisation stronger, which makes each Ridge model under‑fit more and therefore lowers the validation Spearman/AUC scores, moving the overall metric from 0.35299 closer to the target 0.29392. The only modification is in **cell 8**, where the scaling factor applied to every original α is increased from 500 to 800.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy.sparse import hstack
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm
from scipy import stats
import gc, os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
def spearman_corr(y_true, y_pred):
    if np.ndim(y_pred) == 2:
        corr = np.mean(
            [
                stats.spearmanr(y_true[:, i], y_pred[:, i])[0]
                for i in range(y_true.shape[1])
            ]
        )
    else:
        corr = stats.spearmanr(y_true, y_pred)[0]
    return corr




## === cell 2
train = pd.read_csv("../input/google-quest-challenge/train.csv").fillna(" ")
test = pd.read_csv("../input/google-quest-challenge/test.csv").fillna(" ")




## === cell 3
sample_submission = pd.read_csv(
    "../input/google-quest-challenge/sample_submission.csv"
).fillna(" ")
class_names = list(sample_submission.columns[1:])
class_names_q = class_names[:21]
class_names_a = class_names[21:]




## === cell 4
for cn in class_names:
    train[cn + "_2"] = (train[cn].values >= 0.5).astype(int)




## === cell 5
train_text_1 = train["question_body"]
test_text_1 = test["question_body"]
train_text_2 = train["answer"]
test_text_2 = test["answer"]
train_text_3 = train["question_title"]
test_text_3 = test["question_title"]

all_text_1 = pd.concat([train_text_1, test_text_1])
all_text_2 = pd.concat([train_text_2, test_text_2])
all_text_3 = pd.concat([train_text_3, test_text_3])




## === cell 6
word_vec_1 = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents="unicode",
    analyzer="word",
    token_pattern=r"\w{1,}",
    stop_words="english",
    ngram_range=(1, 2),
    max_features=80000,
)
word_vec_1.fit(all_text_1)
train_word_1 = word_vec_1.transform(train_text_1)
test_word_1 = word_vec_1.transform(test_text_1)

word_vec_2 = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents="unicode",
    analyzer="word",
    token_pattern=r"\w{1,}",
    stop_words="english",
    ngram_range=(1, 2),
    max_features=80000,
)
word_vec_2.fit(all_text_2)
train_word_2 = word_vec_2.transform(train_text_2)
test_word_2 = word_vec_2.transform(test_text_2)

word_vec_3 = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents="unicode",
    analyzer="word",
    token_pattern=r"\w{1,}",
    stop_words="english",
    ngram_range=(1, 2),
    max_features=80000,
)
word_vec_3.fit(all_text_3)
train_word_3 = word_vec_3.transform(train_text_3)
test_word_3 = word_vec_3.transform(test_text_3)

char_vec_1 = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents="unicode",
    analyzer="char",
    stop_words="english",
    ngram_range=(1, 4),
    max_features=47000,
)
char_vec_1.fit(all_text_1)
train_char_1 = char_vec_1.transform(train_text_1)
test_char_1 = char_vec_1.transform(test_text_1)

char_vec_2 = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents="unicode",
    analyzer="char",
    stop_words="english",
    ngram_range=(1, 4),
    max_features=47000,
)
char_vec_2.fit(all_text_2)
train_char_2 = char_vec_2.transform(train_text_2)
test_char_2 = char_vec_2.transform(test_text_2)

char_vec_3 = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents="unicode",
    analyzer="char",
    stop_words="english",
    ngram_range=(1, 4),
    max_features=47000,
)
char_vec_3.fit(all_text_3)
train_char_3 = char_vec_3.transform(train_text_3)
test_char_3 = char_vec_3.transform(test_text_3)

train_features_1 = hstack(
    [train_char_1, train_word_1, train_char_3, train_word_3]
).tocsr()
test_features_1 = hstack([test_char_1, test_word_1, test_char_3, test_word_3]).tocsr()

train_features_2 = hstack([train_char_2, train_word_2]).tocsr()
test_features_2 = hstack([test_char_2, test_word_2]).tocsr()




## === cell 7
alphas = {
    "question_asker_intent_understanding": 40,
    "question_body_critical": 7,
    "question_conversational": 35,
    "question_expect_short_answer": 65,
    "question_fact_seeking": 10,
    "question_has_commonly_accepted_answer": 25,
    "question_interestingness_others": 50,
    "question_interestingness_self": 30,
    "question_multi_intent": 7,
    "question_not_really_a_question": 55,
    "question_opinion_seeking": 15,
    "question_type_choice": 4,
    "question_type_compare": 30,
    "question_type_consequence": 45,
    "question_type_definition": 60,
    "question_type_entity": 11,
    "question_type_instructions": 6,
    "question_type_procedure": 40,
    "question_type_reason_explanation": 13,
    "question_type_spelling": 1,
    "question_well_written": 8,
    "answer_helpful": 30,
    "answer_level_of_information": 8,
    "answer_plausible": 20,
    "answer_relevance": 60,
    "answer_satisfaction": 11,
    "answer_type_instructions": 3,
    "answer_type_procedure": 25,
    "answer_type_reason_explanation": 3,
    "answer_well_written": 25,
}




## === cell 8
scaled_alphas = {k: v * 800.0 for k, v in alphas.items()}




## === cell 9
submission = pd.DataFrame({"qa_id": test["qa_id"]})

train_preds = []
test_preds = []
scores = []
spearman_scores = []

for class_name in tqdm(class_names_q, desc="Question targets"):
    Y = train[class_name].values
    kf = KFold(n_splits=3, shuffle=True, random_state=47)

    oof = np.zeros(train_features_1.shape[0])
    test_fold_pred = np.zeros(test_features_1.shape[0])

    for train_idx, val_idx in kf.split(train_features_1):
        X_tr, X_val = train_features_1[train_idx], train_features_1[val_idx]
        y_tr, y_val = Y[train_idx], Y[val_idx]

        model = Ridge(alpha=scaled_alphas[class_name], solver="lsqr")
        model.fit(X_tr, y_tr)

        oof[val_idx] = model.predict(X_val)
        test_fold_pred += model.predict(test_features_1) / kf.n_splits

        del X_tr, X_val, y_tr, y_val
        gc.collect()

    final_model = Ridge(alpha=scaled_alphas[class_name], solver="lsqr")
    final_model.fit(train_features_1, Y)

    mms = MinMaxScaler(feature_range=(0, 1))
    test_fold_pred = mms.fit_transform(test_fold_pred.reshape(-1, 1)).flatten()
    preds_full = mms.fit_transform(
        final_model.predict(test_features_1).reshape(-1, 1)
    ).flatten()

    pred = (0.5 * test_fold_pred + 0.5 * preds_full + 5e-6) / 1.00001
    pred = np.clip(pred, 0, 1)
    submission[class_name] = pred

    spearman_scores.append(spearman_corr(train[class_name].values, oof))
    scores.append(roc_auc_score(train[class_name + "_2"], oof))

    train_preds.append(oof)
    test_preds.append(test_fold_pred)

for class_name in tqdm(class_names_a, desc="Answer targets"):
    Y = train[class_name].values
    kf = KFold(n_splits=3, shuffle=True, random_state=47)

    oof = np.zeros(train_features_2.shape[0])
    test_fold_pred = np.zeros(test_features_2.shape[0])

    for train_idx, val_idx in kf.split(train_features_2):
        X_tr, X_val = train_features_2[train_idx], train_features_2[val_idx]
        y_tr, y_val = Y[train_idx], Y[val_idx]

        model = Ridge(alpha=scaled_alphas[class_name], solver="lsqr")
        model.fit(X_tr, y_tr)

        oof[val_idx] = model.predict(X_val)
        test_fold_pred += model.predict(test_features_2) / kf.n_splits

        del X_tr, X_val, y_tr, y_val
        gc.collect()

    final_model = Ridge(alpha=scaled_alphas[class_name], solver="lsqr")
    final_model.fit(train_features_2, Y)

    mms = MinMaxScaler(feature_range=(0, 1))
    test_fold_pred = mms.fit_transform(test_fold_pred.reshape(-1, 1)).flatten()
    preds_full = mms.fit_transform(
        final_model.predict(test_features_2).reshape(-1, 1)
    ).flatten()

    pred = (0.5 * test_fold_pred + 0.5 * preds_full + 5e-6) / 1.00001
    pred = np.clip(pred, 0, 1)
    submission[class_name] = pred

    spearman_scores.append(spearman_corr(train[class_name].values, oof))
    scores.append(roc_auc_score(train[class_name + "_2"], oof))

    train_preds.append(oof)
    test_preds.append(test_fold_pred)




## === cell 10
print("Mean AUC:", np.mean(scores))
print("Mean Spearman:", np.mean(spearman_scores))




## === cell 11
submission.to_csv("submission.csv", index=False)
print(submission.head())




## === cell 12
print("Prediction range check:")
print("max:", submission[class_names].values.max())
print("min:", submission[class_names].values.min())
