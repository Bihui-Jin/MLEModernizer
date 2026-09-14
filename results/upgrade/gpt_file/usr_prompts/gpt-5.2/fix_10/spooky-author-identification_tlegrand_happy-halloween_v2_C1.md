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
Given some text, predict the author.

## Metric
Multi-class logarithmic loss. 

The submitted probabilities for a given sentences are not required to sum to one because they are rescaled prior to being scored (each row is divided by the row sum).

In order to avoid the extremes of the log function, predicted probabilities are replaced with \\(max(min(p,1-10^{-15}),10^{-15})\\).

## Submission Format
You must submit a csv file with the id, and a probability for each of the three classes. The order of the rows does not matter. The file must have a header and should look like the following:

```
id,EAP,HPL,MWS
id07943,0.33,0.33,0.33
...
```

## Dataset 
### File descriptions
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Data fields
- **id** - a unique identifier for each sentence
- **text** - some text written by one of the authors
- **author** - the author of the sentence (EAP: Edgar Allan Poe, HPL: HP Lovecraft; MWS: Mary Wollstonecraft Shelley)

# 2. Python version

3.6

# 3. Installed packages

geopandas==0.14.4
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
seaborn==0.12.2
sklearn-pandas==2.2.0
wordcloud==1.9.4

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (98 lines)
            sample_submission.csv (1959 lines)
            sample_submission.csv.zip (7.4 kB)
            test.csv (1959 lines)
            test.csv.zip (133.3 kB)
            train.csv (17622 lines)
            train.csv.zip (1.2 MB)
            train.zip (1.2 MB)
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 6 other files
                spooky-author-identification/
        input/
            description.md (98 lines)
            sample_submission.csv (1959 lines)
            sample_submission.csv.zip (7.4 kB)
            test.csv (1959 lines)
            test.csv.zip (133.3 kB)
            train.csv (17622 lines)
            train.csv.zip (1.2 MB)
            train.zip (1.2 MB)
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 6 other files
                spooky-author-identification/
        working/
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 6 other files
                spooky-author-identification/
```

-> data/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> data/spooky-author-identification/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> data/spooky-author-identification/test.csv has 1958 rows and 2 columns.
The columns are: id, text

-> data/spooky-author-identification/train.csv has 17621 rows and 3 columns.
The columns are: id, text, author

-> data/test.csv has 1958 rows and 2 columns.
The columns are: id, text

-> data/train.csv has 17621 rows and 3 columns.
The columns are: id, text, author

-> input/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> (stopped after 10 files for performance)

# 5. Target score

1.1111

# 6. Current score

0.99328

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.43608) has done: 'The timeout is dominated by fitting the pipeline multiple times inside `cross_val_score` with an expensive `FeatureUnion` (CountVectorizer + TfidfVectorizer) and by Python-level `.apply(lambda ...)` feature transformers. I keep the exact same model, features, and evaluation semantics, but: (1) remove the redundant pre-fit before cross-validation, (2) enable multi-core parallelism for cross-validation and for LogisticRegression (still the same solver/logic), and (3) replace the slow `.apply` lambdas with equivalent vectorized pandas string operations to cut feature-extraction overhead. I also avoid rendering heavy plots/wordclouds in the timed run (they don’t affect predictions) while leaving the code structure intact.'
- What this solution (achieved 0.574) has done: 'Your current score (0.43608 logloss) is much better than the target (1.1111), and since lower is better we should *degrade* performance slightly toward the target band with minimal, controlled changes. The smallest safe lever that preserves the same model/feature/training semantics is to increase regularization on the existing LogisticRegression (smaller `C`), which generally worsen logloss without breaking the pipeline. I also switch cross-validation scoring from accuracy to `neg_log_loss` so you can see the metric you actually care about, without affecting the final submission generation. Everything else (FeatureUnion, vectorizers, transformers, training loop, and submission format) remains unchanged.'
- What this solution (achieved 0.77622) has done: 'Your current logloss (0.574) is much better than the target (1.1111), and since lower is better we should intentionally move it upward toward the target band with the smallest safe change. The most controlled lever that preserves the exact same pipeline/features/training semantics is to further strengthen regularization in the existing `LogisticRegression` by reducing `C`, which typically worsens fit/calibration and increases logloss. I keep everything else identical (same FeatureUnion, vectorizers, split, CV metric, and submission formatting) so the run stays stable and still produces a valid `submission.csv`. This should nudge performance toward the target without altering core logic.'
- What this solution (achieved 0.87783) has done: 'Your current logloss (0.77622) is still substantially better (lower) than the target (1.1111), so to move *toward* the target we should intentionally make the model a bit less expressive while preserving the exact same pipeline/features/training semantics. The smallest, most controlled lever here is stronger regularization in the existing `LogisticRegression` (reduce `C`), which typically increases logloss without breaking anything. I also add `random_state` to LogisticRegression for stability/reproducibility (doesn’t change core logic) and keep the submission writing exactly the same. Everything else (FeatureUnion components, vectorizers, CV setup, and probability output) remains unchanged.'
- What this solution (achieved 0.94091) has done: 'Your current logloss (0.87783) is better (lower) than the target (1.1111), so we should intentionally nudge performance downward (higher logloss) with the smallest, controlled change. The safest lever that preserves the exact same pipeline/features/training semantics is to strengthen regularization further by reducing `C` in the existing `LogisticRegression`. I’m also setting `solver="lbfgs"` explicitly to keep behavior stable/reproducible across environments while keeping the same model class and probability outputs. Everything else (FeatureUnion components, CV, fitting on full train, and submission formatting) stays the same and still writes `submission.csv`.'
- What this solution (achieved 0.97846) has done: 'Your current logloss (0.94091) is still better (lower) than the target (1.1111), so we should *intentionally* worsen it slightly to move closer to the target band with the smallest controlled change. The safest lever that preserves the exact same pipeline/features/training semantics is to further strengthen regularization in the existing `LogisticRegression` by reducing `C` a bit. I keep everything else identical (FeatureUnion components, vectorizers, CV setup, training loop, and submission formatting) so the script remains stable and still writes a valid `submission.csv`. I also print CV/holdout logloss as before so you can quickly verify we moved in the right direction before submitting.'
- What this solution (achieved 0.99328) has done: 'Your current logloss (0.97846) is still better (lower) than the target (1.1111), so we should deliberately worsen it slightly to move closer to the target band with the smallest safe change. The most controlled lever that preserves the exact same pipeline/features/training semantics is to strengthen regularization a bit more by reducing `C` in the existing `LogisticRegression`. I keep everything else identical (vectorizers, extra numeric features, CV metric, fitting procedure, and submission formatting) to maintain stability and ensure the script still produces a valid `submission.csv`. This should nudge the public score upward (worse) toward ~1.11 without breaking the run.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud, STOPWORDS

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

np.random.seed(12)

plt.ioff()




## === cell 1
def _resolve_path(fname):
    candidates = [
        os.path.join("/kaggle/input", fname),
        os.path.join("/kaggle/input/spooky-author-identification", fname),
        os.path.join("../input", fname),
        os.path.join("../input/spooky-author-identification", fname),
        os.path.join("./input", fname),
        os.path.join("./kaggle/input", fname),
        os.path.join("./kaggle/data", fname),
        os.path.join("./kaggle/data/spooky-author-identification", fname),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return fname


train_path = _resolve_path("train.csv")
test_path = _resolve_path("test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)



## === cell 2
train.head()
print("--- Shape ---")
print(train.shape)
print("--- Missing values (%) ---")
print(train.isnull().sum() * 100 / len(train))



## === cell 3
if False:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=train, x="author")
    plt.tight_layout()
    plt.show()




## === cell 4
def build_corpus(data):
    data = str(data)
    corpus = ""
    for sent in data:
        corpus += str(sent)
    return corpus




## === cell 5
eap = train[train.author == "EAP"]
hpl = train[train.author == "HPL"]
mws = train[train.author == "MWS"]



## === cell 6
if False:
    plt.figure(figsize=(15, 10))
    plt.subplot(331)
    eap_wc = WordCloud(background_color="white", max_words=100, stopwords=STOPWORDS)
    eap_wc.generate(build_corpus(eap.text))
    plt.title("Edgar Allan Poe", fontsize=20)
    plt.imshow(eap_wc, interpolation="bilinear")
    plt.axis("off")

    plt.subplot(332)
    hpl_wc = WordCloud(background_color="white", max_words=100, stopwords=STOPWORDS)
    hpl_wc.generate(build_corpus(hpl.text))
    plt.title("HP Lovecraft", fontsize=20)
    plt.imshow(hpl_wc, interpolation="bilinear")
    plt.axis("off")

    plt.subplot(333)
    mws_wc = WordCloud(background_color="white", max_words=100, stopwords=STOPWORDS)
    mws_wc.generate(build_corpus(mws.text))
    plt.title("Marry Shelley", fontsize=20)
    plt.imshow(mws_wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.show()



## === cell 7
le = LabelEncoder()
author_encoded = le.fit_transform(train.author)



## === cell 8
seed = 12
X_train, X_test, y_train, y_test = train_test_split(
    train, author_encoded, test_size=0.3, random_state=seed, stratify=author_encoded
)

metric = "neg_log_loss"
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)




## === cell 9
class ColumnExtractor(TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def transform(self, X):
        return X[self.cols]

    def fit(self, X, y=None):
        return self




## === cell 10
class CommaCountTransformer(TransformerMixin):
    def transform(self, X, **transform_params):
        s = X.astype(str)
        arr = s.str.count(",").to_numpy(dtype=np.float64).reshape(-1, 1)
        return arr

    def fit(self, X, y=None, **fit_params):
        return self


class WordCountTransformer(TransformerMixin):
    def transform(self, X, **transform_params):
        s = X.astype(str)
        arr = s.str.split().str.len().to_numpy(dtype=np.float64).reshape(-1, 1)
        return arr

    def fit(self, X, y=None, **fit_params):
        return self


class LengthTransformer(TransformerMixin):
    def transform(self, X, **transform_params):
        s = X.astype(str)
        arr = s.str.len().to_numpy(dtype=np.float64).reshape(-1, 1)
        return arr

    def fit(self, X, y=None, **fit_params):
        return self




## === cell 11
pipeline = Pipeline(
    [
        ("extract_texts", ColumnExtractor("text")),
        (
            "features",
            FeatureUnion(
                [
                    ("comma_count", CommaCountTransformer()),
                    ("word_count", WordCountTransformer()),
                    ("text_length", LengthTransformer()),
                    ("count_vect", CountVectorizer()),
                    ("tf_idf", TfidfVectorizer()),
                ]
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                n_jobs=-1,
                multi_class="auto",
                random_state=12,
                solver="lbfgs",
                C=0.00011,  # was 0.00015
            ),
        ),
    ]
)



## === cell 12
score_pipe = cross_val_score(
    pipeline, X_train, y_train, cv=kfold, scoring=metric, n_jobs=-1
)
print("CV mean logloss = %.4f, std = %.4f" % (-np.mean(score_pipe), np.std(score_pipe)))

clf_pipe = pipeline.fit(X_train, y_train)
from sklearn.metrics import log_loss

holdout_proba = clf_pipe.predict_proba(X_test)
print("Holdout logloss = %.4f" % (log_loss(y_test, holdout_proba),))



## === cell 13
if False:
    conf_mat = confusion_matrix(y_test, clf_pipe.predict(X_test))
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues")
    plt.xticks(range(3), ("EAP", "HPL", "MWS"), horizontalalignment="left")
    plt.yticks(range(3), ("EAP", "HPL", "MWS"), rotation=0)
    plt.tight_layout()
    plt.show()



## === cell 14
final_clf = pipeline.fit(train, author_encoded)

target_names = ["EAP", "HPL", "MWS"]
proba = final_clf.predict_proba(test)

proba_df = pd.DataFrame(proba, columns=list(le.classes_))
proba_df = proba_df[target_names]

submission = pd.concat(
    [test[["id"]].reset_index(drop=True), proba_df.reset_index(drop=True)], axis=1
)
submission_path = "./submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission.head())
