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

0.46668

# 6. Current score

2.85143

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.55786) has done: 'Diagnosis: The crash happens in cell 14 because `pandas.concat` in pandas 2.x no longer accepts the `axis` argument positionally; `pd.concat([..], 1)` is treated as passing a second positional argument, raising `TypeError: concat() takes 1 positional argument but 2 were given`. The fix is to pass `axis=1` as a keyword argument. This preserves the exact submission dataframe structure expected (id + 3 probability columns) without changing any modeling logic.

Patch summary: In cell 14, change `pd.concat([test["id"], y_pred], 1)` to `pd.concat([test["id"], y_pred], axis=1)` to be compatible with pandas 2.2.3.

Updated cells:'
- What this solution (achieved 3.18337) has done: 'Your current score (0.55786, lower-is-better) is worse than the target (0.46668), so we should improve the model slightly while keeping the same overall pipeline idea. The biggest score issue is that `MultinomialNB` expects non‑negative feature values, but your `FeatureUnion` mixes in numeric count features alongside a `TfidfVectorizer` that (by default) can apply L2 normalization; this combination can lead to poorly calibrated probabilities and worse log loss. A minimal, core-logic-preserving fix is to keep the exact same feature blocks and classifier, but disable TF‑IDF normalization (`norm=None`) to make feature scales more compatible with Naive Bayes and generally improve log-loss stability. I also switch the validation metric from accuracy to `neg_log_loss` for the printed CV diagnostics (this does not change training/predictions), and keep your submission formatting intact.'
- What this solution (achieved 3.18337) has done: 'Your current log loss (3.18337, lower-is-better) is far worse than the target (0.46668), so we should apply a small, legitimate correction that typically fixes catastrophic log-loss: ensure the predicted probability columns map to the correct author labels. Right now you assume `["EAP","HPL","MWS"]` matches the internal class order, but `LabelEncoder` sorts labels alphabetically and `MultinomialNB.predict_proba` returns columns in `clf_pipe.classes_` order, so your submission can be mislabeled, which explodes log loss. I keep your exact model/pipeline and only change the submission-building step to derive column names from `le.inverse_transform(clf_pipe.classes_)`, then reindex to `["EAP","HPL","MWS"]` to match the required submission schema. This should move the score sharply toward the target without changing training logic.'
- What this solution (achieved 3.18337) has done: 'Your score is far worse than the target (3.18337 vs 0.46668, lower-is-better), which is consistent with a submission that has misaligned probability columns and/or NaN probabilities after reindexing. I keep your exact model/pipeline, but make the submission-building step robust by (1) ensuring the mapped class-name columns are correct and (2) filling any missing columns with a tiny epsilon so no NaNs/zeros cause catastrophic log loss. I also explicitly cast `test["id"]` to a Series and enforce final column order exactly as required. These are minimal changes focused purely on producing valid, correctly-aligned probabilities and should move the score sharply toward the target band without changing training logic.'
- What this solution (achieved 3.1903) has done: 'Your current score is far worse than the target (3.18337 vs 0.46668, lower-is-better), which is most consistent with the submission probabilities not corresponding to the correct class columns at scoring time. I keep your exact pipeline/model and only (1) stop using `LabelEncoder` for model training so the model is trained directly on the correct string labels, and (2) build the submission columns directly from `clf_pipe.classes_` and then reindex to `["EAP","HPL","MWS"]`. This is a minimal change that preserves the core logic but removes the class-mapping failure mode that can catastrophically inflate log loss. I also keep the tiny epsilon fill to avoid any accidental missing columns/NaNs from causing log(0) issues.'
- What this solution (achieved 2.94375) has done: 'Your score (3.1903, lower-is-better) is far from the target (0.46668), which most commonly indicates the submission probabilities are not aligned to the correct rows/ids or contain invalid values. I keep your exact feature pipeline and MultinomialNB model, but (1) fit the model on the full training data before predicting test (so predictions aren’t from a 70% subset), and (2) build the submission by starting from `sample_submission.csv` to guarantee row alignment and column order, then fill it using an explicit id→probability mapping. I also add a tiny probability floor to avoid any accidental exact zeros (which can catastrophically inflate log loss) without changing the model itself. These are minimal, high-impact fixes focused on producing a correctly formatted, correctly aligned submission and moving log-loss sharply toward the target band.'
- What this solution (achieved 2.94375) has done: 'Your current log loss (2.94375) is much worse than the target (0.46668), and the most likely cause (given your pipeline is otherwise reasonable for this competition) is that the numeric feature blocks output integer/float DataFrames that can include zeros but also can interact poorly with sparse text features, and more importantly that your combined `FeatureUnion` is producing a mixed dense+CSR output that can behave inconsistently across pandas/sklearn versions. I make the smallest change that stabilizes feature union output for MultinomialNB by ensuring the three numeric transformers return non-negative numpy arrays with explicit 2D shape (not DataFrames), which prevents accidental object dtype / alignment issues that can severely harm probability estimates (and log loss). I also keep your class-column alignment logic, but add a single safety renormalization of each row (allowed by the metric, since Kaggle rescales anyway) to avoid pathological row-sum scaling due to numeric-feature dominance. These changes preserve your model, features, and training approach, while targeting the failure modes that lead to catastrophically bad log loss.'
- What this solution (achieved 0.73276) has done: 'Your score is much worse than the target (2.94375 vs 0.46668, lower-is-better), which strongly suggests the submission probabilities are being produced from a model that isn’t learning meaningful signal due to an invalid feature setup for `MultinomialNB`. The main issue is that `TfidfVectorizer` produces fractional TF‑IDF weights and (with `sublinear_tf`/idf) values that are not “counts”, which commonly hurts MultinomialNB log-loss; a minimal, core-logic-preserving fix is to keep the exact same Naive Bayes classifier and feature union, but make the text feature block strictly count-based (CountVectorizer + TfidfTransformer with `use_idf=False`) so NB’s assumptions fit better. I also keep your existing submission alignment/renormalization logic, but ensure the combined matrix stays non-negative and sparse-friendly. These changes should improve log loss substantially toward the target while keeping the same overall pipeline design (FeatureUnion -> MultinomialNB) and producing a valid submission CSV.'
- What this solution (achieved 1.07498) has done: 'Your current log loss (0.73276, lower-is-better) is still worse than the target (0.46668), so we should make a small, low-risk improvement without changing the overall FeatureUnion→MultinomialNB approach. The most impactful minimal tweak for MultinomialNB on this competition is to improve tokenization/coverage by using word n-grams and basic accent stripping, which typically lowers log loss while preserving the same model family and training flow. I only adjust the two text vectorizers inside your existing FeatureUnion (CountVectorizer parameters) and keep everything else (numeric features, classifier, fitting, submission alignment) the same. This should move the score closer to the target band without introducing new training schemes or changing the submission semantics.'
- What this solution (achieved 2.81423) has done: 'Your current log loss (1.07498, lower-is-better) is still worse than the target (0.46668), so we should make a small improvement that keeps the exact same FeatureUnion→MultinomialNB pipeline and training flow. The lowest-risk boost for this competition is to add character n-grams as an additional text feature block (still count-based, still non-negative), which typically improves author identification log loss without changing the model family. I also remove the redundant duplicate word CountVectorizer by keeping one word-count block and one char-count block to avoid overweighting the same signal. Submission construction, class-column alignment, clipping, and row-wise renormalization remain intact.'
- What this solution (achieved 2.82237) has done: 'Your score is far worse than the target, so the most likely remaining issue is “feature dominance”: the three raw numeric features (length/word count/comma count) can overwhelm the sparse text counts for MultinomialNB, leading to poorly calibrated probabilities and high log loss. I keep the exact same FeatureUnion→MultinomialNB pipeline and training flow, but rescale those three numeric features down by a small constant so they remain informative without swamping the text features. This is a minimal, local change that preserves the model family/logic while typically improving log-loss calibration. Submission construction stays the same (class order, clipping, and id alignment).'
- What this solution (achieved 2.85133) has done: 'Your current log loss (2.82237) is far worse than the target (0.46668), and the most likely cause left (given you already fixed class-column alignment and id alignment) is feature miscalibration in `MultinomialNB` due to heavy redundancy: you include multiple overlapping word-count blocks plus numeric features, which can distort NB’s likelihoods and produce overconfident wrong probabilities (bad for log loss). I keep the exact same overall core logic (FeatureUnion → MultinomialNB, same transformers, same training flow), but make a minimal change by removing the redundant duplicate word CountVectorizer inside the `tf_idf_noidf` sub-pipeline and reuse a single shared word-count representation. This typically improves probability calibration (and log loss) without changing the model family or adding new training tricks. Submission writing stays the same (sample_submission merge, correct columns, clipping, and row normalization).'
- What this solution (achieved 2.85143) has done: 'Your score is much worse than the target (2.85133 vs 0.46668, lower-is-better), so we should make a small change that typically improves MultinomialNB log-loss without changing the overall FeatureUnion→MultinomialNB approach. The numeric engineered features can dominate sparse text counts and distort NB likelihoods, so I remove those three numeric blocks (comma/word/length) from the FeatureUnion while keeping the same text feature blocks and classifier. This is a minimal change in feature composition (not model/training loop) and is very likely to move log loss substantially closer to the target. I keep your existing class-column alignment, clipping, row renormalization, and sample_submission-based id alignment to ensure a valid submission.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
from string import punctuation
from nltk import pos_tag
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist
from wordcloud import WordCloud, STOPWORDS

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix



## === cell 1
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sample_sub = pd.read_csv("../input/sample_submission.csv")



## === cell 2
train.head()
print("--- Shape ---")
print(train.shape)
print("--- Missing values ---")
train.isnull().sum() * 100 / len(train)



## === cell 3
sns.countplot(x=train["author"])




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



## === cell 7
le = LabelEncoder()
_ = le.fit(
    train.author
)  # kept for compatibility with the rest of the notebook, but not used for training

y_labels = train.author.astype(str)

seed = 12
X_train, X_test, y_train, y_test = train_test_split(
    train.text, y_labels, test_size=0.3, random_state=seed, stratify=y_labels
)

metric = "neg_log_loss"
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)




## === cell 8
class ColumnExtractor(TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def transform(self, X):
        Xcols = X[self.cols]
        return Xcols

    def fit(self, X, y=None):
        return self


class ModelTransformer(TransformerMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return self

    def transform(self, X, **transform_params):
        return pd.DataFrame(self.model.predict(X))




## === cell 9
NUMERIC_SCALE = 0.01


class CommaCountTransformer(TransformerMixin):
    def transform(self, X, **transform_params):
        arr = (
            X.apply(lambda x: str(x).count(","))
            .to_numpy(dtype=np.float64)
            .reshape(-1, 1)
        )
        return arr * NUMERIC_SCALE

    def fit(self, X, y=None, **fit_params):
        return self


class WordCountTransformer(TransformerMixin):
    def transform(self, X, **transform_params):
        arr = (
            X.apply(lambda x: len(str(x).split()))
            .to_numpy(dtype=np.float64)
            .reshape(-1, 1)
        )
        return arr * NUMERIC_SCALE

    def fit(self, X, y=None, **fit_params):
        return self


class LengthTransformer(TransformerMixin):
    def transform(self, X, **transform_params):
        arr = X.apply(lambda x: len(str(x))).to_numpy(dtype=np.float64).reshape(-1, 1)
        return arr * NUMERIC_SCALE

    def fit(self, X, y=None, **fit_params):
        return self




## === cell 10
word_cv_params = dict(
    lowercase=False,  # preserve your original casing behavior
    strip_accents="unicode",  # improves token consistency
    ngram_range=(1, 2),  # word unigrams+bigrams
    min_df=2,  # reduce one-off noise features
)

char_cv_params = dict(
    lowercase=False,  # keep consistent with word block
    strip_accents="unicode",
    analyzer="char_wb",  # character n-grams within word boundaries (strong for author style)
    ngram_range=(3, 5),
    min_df=2,
)

pipeline = Pipeline(
    [
        (
            "features",
            FeatureUnion(
                [
                    ("word_count_vect", CountVectorizer(**word_cv_params)),
                    ("char_count_vect", CountVectorizer(**char_cv_params)),
                    (
                        "tf_idf_noidf",
                        Pipeline(
                            [
                                (
                                    "cv",
                                    CountVectorizer(
                                        lowercase=False,
                                        strip_accents="unicode",
                                        ngram_range=(1, 1),
                                        min_df=2,
                                    ),
                                ),
                                ("tf", TfidfTransformer(use_idf=False, norm=None)),
                            ]
                        ),
                    ),
                ]
            ),
        ),
        ("classifier", MultinomialNB()),
    ]
)



## === cell 11
clf_pipe = pipeline.fit(X_train, y_train)
score_pipe = cross_val_score(clf_pipe, X_train, y_train, cv=kfold, scoring=metric)
print(
    "Mean score = %.5f, Std deviation = %.5f"
    % (np.mean(score_pipe), np.std(score_pipe))
)

score_pipe_test = clf_pipe.score(X_test, y_test)
print("Holdout accuracy (unchanged reporting) = %.5f" % (score_pipe_test))



## === cell 12
conf_mat = confusion_matrix(
    y_test, clf_pipe.predict(X_test), labels=["EAP", "HPL", "MWS"]
)
sns.heatmap(conf_mat, annot=True)
plt.xticks(range(3), ("EAP", "HPL", "MWS"), horizontalalignment="left")
plt.yticks(range(3), ("EAP", "HPL", "MWS"), rotation=0)



## === cell 13
clf_pipe_full = pipeline.fit(train.text, train.author.astype(str))

required_order = ["EAP", "HPL", "MWS"]
eps = 1e-15  # match evaluation clipping scale

proba = clf_pipe_full.predict_proba(test.text)
model_class_names = list(clf_pipe_full.classes_)
y_pred = pd.DataFrame(proba, columns=model_class_names)

for col in required_order:
    if col not in y_pred.columns:
        y_pred[col] = eps
y_pred = y_pred.reindex(columns=required_order)

y_pred = y_pred.clip(lower=eps)
row_sums = y_pred.sum(axis=1).to_numpy()
row_sums[row_sums == 0.0] = 1.0
y_pred = y_pred.div(row_sums, axis=0).clip(lower=eps)

submission = sample_sub[["id"]].copy()
submission["id"] = submission["id"].astype(str)

pred_with_id = pd.concat(
    [test["id"].astype(str).reset_index(drop=True), y_pred.reset_index(drop=True)],
    axis=1,
)
pred_with_id.columns = ["id"] + required_order

submission = submission.merge(pred_with_id, on="id", how="left", validate="one_to_one")

for col in required_order:
    if col not in submission.columns:
        submission[col] = eps
submission[required_order] = submission[required_order].fillna(eps).clip(lower=eps)

submission = submission[["id"] + required_order]
submission.to_csv("./submission.csv", index=False)
print(
    "Wrote submission:", os.path.abspath("./submission.csv"), "shape=", submission.shape
)
print(submission.head())
