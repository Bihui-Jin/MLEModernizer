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
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
nltk==3.9.2
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
tf_keras==2.18.0
wordcloud==1.9.4

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

0.19412

# 6. Current score

0.2306

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.25493) has done: 'I fix the two runtime blockers: the `stopwords` lookup that fails because the NLTK corpus isn’t available in this environment, and the Seaborn `barplot` calls that now require keyword arguments. These changes are score-neutral and only affect EDA/feature engineering stability. I also make the TF-IDF+SVD pipeline deterministic and correct a subtle logic issue where the same `TruncatedSVD` instance was reused across different text fields (overwriting its fitted state), which can otherwise cause inconsistent transforms; this preserves the intended architecture and typically improves stability (and should not materially push score away from your already-better-than-target result). Finally, I ensure the script always writes a valid `submission.csv` with predictions clipped to `[0,1]`.'
- What this solution (achieved 0.24806) has done: 'I fix the runtime blocker happening at import time (`AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`) by avoiding the standalone `keras` package (which pulls in protobuf-incompatible deps here) and using `tf_keras` instead, keeping the exact same model/fit loop and layers. I also make the seaborn plotting calls compatible with newer seaborn by using keyword arguments, but keep EDA code functionally identical. Since your current score (0.25493) is already above the target (0.19412) and within the ±10% band, I won’t make any score-changing modeling/training tweaks—only stability fixes. The script still write a valid `submission.csv` with predictions clipped to `[0,1]`.'
- What this solution (achieved 0.25726) has done: 'I fix the import-time crash that prevents the notebook from running by avoiding the `tf_keras`/protobuf incompatibility and using `tensorflow.keras` instead (same Keras API, same model definition and training loop). I also update the Seaborn `barplot` calls to use keyword arguments so the EDA cells don’t error on newer seaborn, without affecting modeling. Because your current score (0.24806) is already above the target (0.19412) and outside the ±10% target band but on the “better-than-target” side, I not make any modeling/training changes that could move score further away; the patch is intended to be score-neutral. The script still always write a valid `submission.csv` with the correct columns and predictions clipped to `[0,1]`.'
- What this solution (achieved 0.25345) has done: 'I fix the import-time crash (`MessageFactory.GetPrototype`) by forcing TensorFlow/Keras to use the legacy `tf_keras` implementation, which avoids the protobuf incompatibility in this Kaggle image while keeping the exact same Keras API, model, and training loop. I also make the seaborn `barplot` calls compatible with seaborn 0.12+ by using keyword arguments (score-neutral, just prevents runtime errors in EDA cells). Since your current score (0.25726) is already above the target (0.19412) and the goal is “toward target” rather than maximizing, I not change any modeling/training settings that would intentionally move the score further away. The script still run end-to-end and always write a valid `submission.csv` with predictions clipped to `[0,1]` and correct columns.'
- What this solution (achieved 0.25075) has done: 'I fix the import-time crash (`MessageFactory.GetPrototype`) by forcing TensorFlow to use the legacy `tf_keras` implementation reliably, without changing the model architecture or training loop. I also make the Seaborn `barplot` calls compatible with seaborn 0.12+ (they now require keyword arguments), which is score-neutral but prevents runtime errors in the EDA cells. Since your current score (0.25345) is already better than the target (0.19412) and we’re aiming “toward target” rather than maximizing, I won’t make any modeling/training changes that would intentionally alter the score. The script run end-to-end and always write a valid `submission.csv` with the correct columns and predictions clipped to `[0,1]`.'
- What this solution (achieved 0.25808) has done: 'I fix the import-time crash (`MessageFactory.GetPrototype`) by avoiding the standalone Keras/protobuf path entirely and using `tensorflow.keras` (same API, same model architecture and training loop). I also make the Seaborn `barplot` calls compatible with seaborn 0.12+ by using keyword arguments, preventing EDA cells from failing without affecting training. These changes are intended to be score-neutral (your current score is already better than the target, so we avoid any model/training changes). Finally, I keep the submission writing intact and ensure predictions are clipped to `[0,1]` and saved as `submission.csv`.'
- What this solution (achieved 0.25435) has done: 'I fix the import-time crash caused by the known protobuf/Keras incompatibility by forcing TensorFlow to use the legacy `tf_keras` implementation, without changing the model architecture or training loop. I also update the seaborn `barplot` calls to use keyword arguments so the EDA cells don’t error on seaborn 0.12+, which is score-neutral. Since your current score (0.25808) is already better than the target (0.19412) and we’re aiming “toward target” (not maximize), I won’t make modeling/training changes that would intentionally alter performance. Finally, I keep the submission writing intact and ensure the output is a valid `submission.csv` with predictions clipped to `[0,1]`.'
- What this solution (achieved 0.24915) has done: 'I fix the import-time crash (`MessageFactory.GetPrototype`) by avoiding the standalone `tensorflow.keras` path in this environment and instead using the compatible `tf_keras` package (same Keras API, same model/layers/training loop). I also update the Seaborn `barplot` calls to use keyword arguments (`x=`/`y=`) so the EDA cells don’t error on seaborn 0.12+. Since your current score (0.25435) is already above the target (0.19412), I not change any modeling, features, folds, epochs, or post-processing beyond stability; this should be score-neutral. The script run end-to-end and always write a valid `submission.csv` with the correct columns and predictions clipped to `[0,1]`.'
- What this solution (achieved 0.09777) has done: 'I fix the import-time crash (`AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`) by avoiding TensorFlow/Keras imports entirely since this solution uses a simple MLP and can be trained with scikit-learn without changing the feature pipeline. To keep the core learning approach (a multi-output feedforward neural network trained per fold) essentially the same while unblocking runtime, I replace the Keras `Sequential` MLP with an equivalent `sklearn.neural_network.MLPRegressor` inside the same KFold loop and keep sigmoid-bounded outputs via clipping to `[0,1]`. I also make the seaborn `barplot` calls compatible with seaborn 0.12+ (keyword args) and ensure all TF-IDF inputs are strings (fillna) to prevent sporadic vectorizer errors. Because your current score is already better than the target and the goal is “toward target”, I avoid any extra modeling/tuning changes beyond what’s required for stability and a valid `submission.csv`.'
- What this solution (achieved 0.2306) has done: 'Your current score (0.09777) is well below the target (0.19412), so we should make a small, legitimate improvement without changing the overall pipeline (TF‑IDF+SVD features + fold-based MLP training + Spearman eval). The biggest issue hurting the MLP here is that SVD components and engineered count features are on very different scales; adding a `StandardScaler` (fit only on each fold’s training split) typically improves convergence and ranking quality while preserving the same modeling approach. I also make the TF‑IDF/SVD step deterministic and safe by limiting `n_components` to the maximum allowed by each matrix shape (prevents silent instability/crashes on small feature spaces) and using `algorithm="randomized"` with fixed `random_state`. Finally, I keep the exact same submission writing and clipping semantics, ensuring a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import gc
import string
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from wordcloud import WordCloud

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor

from scipy.stats import spearmanr

import nltk
from nltk.corpus import stopwords

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

try:
    eng_stopwords = set(stopwords.words("english"))
except Exception:
    try:
        nltk.download("stopwords", quiet=True)
        eng_stopwords = set(stopwords.words("english"))
    except Exception:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        eng_stopwords = set(ENGLISH_STOP_WORDS)

RANDOM_STATE = 666
np.random.seed(RANDOM_STATE)



## === cell 1
train_df = pd.read_csv("/kaggle/input/google-quest-challenge/train.csv")
sample_sub_df = pd.read_csv(
    "/kaggle/input/google-quest-challenge/sample_submission.csv"
)
test_df = pd.read_csv("/kaggle/input/google-quest-challenge/test.csv")



## === cell 2
pd.set_option("display.max_columns", None)
train_df.head()



## === cell 3
test_df.head()



## === cell 4
sample_sub_df.head()



## === cell 5
print(f"Sahpe of training set: {train_df.shape}")
print(f"Sahpe of testing set: {test_df.shape}")



## === cell 6
train_df.columns



## === cell 7
sns.set(rc={"figure.figsize": (11, 8)})
sns.set(style="whitegrid")



## === cell 8
total = len(train_df)



## === cell 9
vc = train_df["category"].value_counts()
ax = sns.barplot(x=vc.index, y=vc.values)
ax.set(xlabel="Category", ylabel="# of records", title="Category vs. # of records")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
for p in ax.patches:
    height = p.get_height()
    ax.text(
        p.get_x() + p.get_width() / 2.0,
        height + 5,
        "{:1.2f}%".format(height / total * 100),
        ha="center",
        fontsize=15,
    )
plt.show()



## === cell 10
v = np.vectorize(lambda x: x.split(".")[0])
sns.set(rc={"figure.figsize": (15, 8)})
vh = train_df["host"].value_counts()
ax = sns.barplot(x=v(vh.index.values), y=vh.values)
ax.set(
    xlabel="Host platforms",
    ylabel="# of records",
    title="Host platforms vs. # of records",
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=50, ha="right")
plt.show()



## === cell 11
wc = WordCloud(background_color="white", max_font_size=85, width=700, height=350)
wc.generate(",".join(train_df["question_title"].astype(str).tolist()))
plt.figure(figsize=(15, 10))
plt.axis("off")
plt.imshow(wc, interpolation="bilinear")



## === cell 12
wc.generate(
    ",".join(train_df["question_body"].astype(str).tolist())
    .replace("gt", "")
    .replace("lt", "")
)
plt.figure(figsize=(15, 10))
plt.axis("off")
plt.imshow(wc, interpolation="bilinear")



## === cell 13
wc.generate(
    ",".join(train_df["answer"].astype(str).tolist())
    .replace("gt", "")
    .replace("lt", "")
)
plt.figure(figsize=(15, 10))
plt.axis("off")
plt.imshow(wc, interpolation="bilinear")



## === cell 14
target_cols = sample_sub_df.drop(["qa_id"], axis=1).columns.values
target_cols



## === cell 15
X_train = train_df.drop(np.concatenate([target_cols, np.array(["qa_id"])]), axis=1)
Y_train = train_df[target_cols]



## === cell 16
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of Y_train: {Y_train.shape}")



## === cell 17
X_train.head()



## === cell 18
X_test = test_df
del test_df
gc.collect()



## === cell 19
for df in (X_train, X_test):
    for c in ["question_title", "question_body", "answer", "category", "host"]:
        df[c] = df[c].fillna("").astype(str)

X_train["answer_size"] = X_train["answer"].apply(lambda x: len(str(x).split()))
X_test["answer_size"] = X_test["answer"].apply(lambda x: len(str(x).split()))

X_train["question_body_size"] = X_train["question_body"].apply(
    lambda x: len(str(x).split())
)
X_test["question_body_size"] = X_test["question_body"].apply(
    lambda x: len(str(x).split())
)

X_train["question_title_size"] = X_train["question_title"].apply(
    lambda x: len(str(x).split())
)
X_test["question_title_size"] = X_test["question_title"].apply(
    lambda x: len(str(x).split())
)

X_train["answer_num_unique_words"] = X_train["answer"].apply(
    lambda x: len(set(str(x).split()))
)
X_test["answer_num_unique_words"] = X_test["answer"].apply(
    lambda x: len(set(str(x).split()))
)

X_train["question_body_num_unique_words"] = X_train["question_body"].apply(
    lambda x: len(set(str(x).split()))
)
X_test["question_body_num_unique_words"] = X_test["question_body"].apply(
    lambda x: len(set(str(x).split()))
)

X_train["answer_num_chars"] = X_train["answer"].apply(lambda x: len(str(x)))
X_test["answer_num_chars"] = X_test["answer"].apply(lambda x: len(str(x)))

X_train["question_body_num_chars"] = X_train["question_body"].apply(
    lambda x: len(str(x))
)
X_test["question_body_num_chars"] = X_test["question_body"].apply(lambda x: len(str(x)))

X_train["answer_num_stopwords"] = X_train["answer"].apply(
    lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords])
)
X_test["answer_num_stopwords"] = X_test["answer"].apply(
    lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords])
)

X_train["question_body_num_stopwords"] = X_train["question_body"].apply(
    lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords])
)
X_test["question_body_num_stopwords"] = X_test["question_body"].apply(
    lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords])
)

X_train["answer_num_punctuations"] = X_train["answer"].apply(
    lambda x: len([c for c in str(x) if c in string.punctuation])
)
X_test["answer_num_punctuations"] = X_test["answer"].apply(
    lambda x: len([c for c in str(x) if c in string.punctuation])
)

X_train["question_body_num_punctuations"] = X_train["question_body"].apply(
    lambda x: len([c for c in str(x) if c in string.punctuation])
)
X_test["question_body_num_punctuations"] = X_test["question_body"].apply(
    lambda x: len([c for c in str(x) if c in string.punctuation])
)

X_train["answer_num_words_upper"] = X_train["answer"].apply(
    lambda x: len([w for w in str(x).split() if w.isupper()])
)
X_test["answer_num_words_upper"] = X_test["answer"].apply(
    lambda x: len([w for w in str(x).split() if w.isupper()])
)

X_train["question_body_num_words_upper"] = X_train["question_body"].apply(
    lambda x: len([w for w in str(x).split() if w.isupper()])
)
X_test["question_body_num_words_upper"] = X_test["question_body"].apply(
    lambda x: len([w for w in str(x).split() if w.isupper()])
)

X_train["answer_num_words_title"] = X_train["answer"].apply(
    lambda x: len([w for w in str(x).split() if w.istitle()])
)
X_test["answer_num_words_title"] = X_test["answer"].apply(
    lambda x: len([w for w in str(x).split() if w.istitle()])
)

X_train["question_body_num_words_title"] = X_train["question_body"].apply(
    lambda x: len([w for w in str(x).split() if w.istitle()])
)
X_test["question_body_num_words_title"] = X_test["question_body"].apply(
    lambda x: len([w for w in str(x).split() if w.istitle()])
)



## === cell 20
X_train.head()



## === cell 21
X_train = X_train.drop(
    [
        "question_user_name",
        "question_user_page",
        "answer_user_name",
        "answer_user_page",
        "url",
    ],
    axis=1,
)
X_test = X_test.drop(
    [
        "question_user_name",
        "question_user_page",
        "answer_user_name",
        "answer_user_page",
        "url",
        "qa_id",
    ],
    axis=1,
)




## === cell 22
def tfidf_svd(train_text, test_text, n_components=1000, random_state=RANDOM_STATE):
    tfv_local = TfidfVectorizer(
        min_df=3,
        max_features=None,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\w{1,}",
        ngram_range=(1, 3),
        use_idf=1,
        smooth_idf=1,
        sublinear_tf=1,
        stop_words="english",
    )
    tr = tfv_local.fit_transform(train_text)
    te = tfv_local.transform(test_text)

    max_comp = min(n_components, tr.shape[0] - 1, tr.shape[1] - 1)
    max_comp = max(1, int(max_comp))

    svd = TruncatedSVD(
        n_components=max_comp,
        random_state=random_state,
        algorithm="randomized",
        n_iter=7,
    )
    tr_s = svd.fit_transform(tr)
    te_s = svd.transform(te)
    return tr_s, te_s


question_title, question_title_test = tfidf_svd(
    X_train["question_title"].values, X_test["question_title"].values, n_components=1000
)
question_body, question_body_test = tfidf_svd(
    X_train["question_body"].values, X_test["question_body"].values, n_components=1000
)
answer, answer_test = tfidf_svd(
    X_train["answer"].values, X_test["answer"].values, n_components=1000
)



## === cell 23
cat_le = LabelEncoder()
cat_le.fit(X_train["category"])
category = cat_le.transform(X_train["category"])
category_test = cat_le.transform(X_test["category"])



## === cell 24
host_le = LabelEncoder()
host_le.fit(pd.concat([X_train["host"], X_test["host"]], ignore_index=True))
host = host_le.transform(X_train["host"])
host_test = host_le.transform(X_test["host"])



## === cell 25
meta_features_train = X_train.drop(
    ["question_title", "question_body", "answer", "category", "host"], axis=1
).to_numpy()
meta_features_test = X_test.drop(
    ["question_title", "question_body", "answer", "category", "host"], axis=1
).to_numpy()



## === cell 26
X_train = np.concatenate([question_title, question_body, answer], axis=1)
X_test = np.concatenate([question_title_test, question_body_test, answer_test], axis=1)



## === cell 27
del question_title
del question_title_test
del answer
del answer_test
del question_body
del question_body_test
gc.collect()



## === cell 28
X_train = np.column_stack((X_train, category, host, meta_features_train))
X_test = np.column_stack((X_test, category_test, host_test, meta_features_test))



## === cell 29
del category
del host
del meta_features_train
del category_test
del host_test
del meta_features_test
gc.collect()



## === cell 30
print(X_train.shape)
print(X_test.shape)



## === cell 31
np.isnan(X_train).any()



## === cell 32
len(X_test)



## === cell 33
folds = 5
seed = 666

kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
test_preds = np.zeros((len(X_test), len(target_cols)), dtype=np.float32)
fold_scores = []

for train_index, val_index in kf.split(X_train):
    x_train = X_train[train_index, :].astype(np.float32, copy=False)
    y_train = Y_train.iloc[train_index].to_numpy(dtype=np.float32)
    x_val = X_train[val_index, :].astype(np.float32, copy=False)
    y_val = Y_train.iloc[val_index]

    scaler = StandardScaler(with_mean=True, with_std=True)
    x_train_s = scaler.fit_transform(x_train)
    x_val_s = scaler.transform(x_val)
    x_test_s = scaler.transform(X_test.astype(np.float32, copy=False))

    model = MLPRegressor(
        hidden_layer_sizes=(256, 128, 128),
        activation="relu",
        solver="adam",
        alpha=0.0,  # keep as close as possible to original (no explicit L2 was used)
        batch_size="auto",
        learning_rate="constant",
        learning_rate_init=0.001,
        max_iter=30,
        shuffle=True,
        random_state=seed,
        early_stopping=False,
        n_iter_no_change=200,  # irrelevant since early_stopping=False; kept explicit for clarity
        verbose=True,
    )

    model.fit(x_train_s, y_train)

    preds = model.predict(x_val_s).astype(np.float32)
    preds = np.clip(preds, 0.0, 1.0)

    overall_score = 0.0
    for col_index, col in enumerate(target_cols):
        overall_score += spearmanr(
            preds[:, col_index], y_val[col].values
        ).correlation / len(target_cols)

    fold_scores.append(overall_score)

    fold_test = model.predict(x_test_s).astype(np.float32)
    fold_test = np.clip(fold_test, 0.0, 1.0)
    test_preds += fold_test / folds

print(fold_scores)



## === cell 34
test_preds = np.clip(test_preds, 0.0, 1.0)

for col_index, col in enumerate(target_cols):
    sample_sub_df[col] = test_preds[:, col_index]

sample_sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sample_sub_df.shape)
