# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Predict the score of student essays.

## Metric
Quadratic weighted kappa.

## Submission Format
For each `essay_id` in the test set, you must predict the corresponding `score` (between 1-6, see [rubric](https://storage.googleapis.com/kaggle-forum-message-attachments/2733927/20538/Rubric_%20Holistic%20Essay%20Scoring.pdf) for more details). The file should contain a header and have the following format:

```
essay_id,score
000d118,3
000fe60,3
001ab80,4
...
```

## Dataset
- **train.csv** - Essays and scores to be used as training data.
    - `essay_id` - The unique ID of the essay
    - `full_text` - The full essay response
    - `score` - Holistic score of the essay on a 1-6 scale
- **test.csv** - The essays to be used as test data. Contains the same fields as `train.csv`, aside from exclusion of `score`.
- **sample_submission.csv** - A submission file in the correct format.
    - `essay_id` - The unique ID of the essay
    - `score` - The predicted holistic score of the essay on a 1-6 scale

# 2. Python version

3.12

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
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        input/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        working/
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
```

-> data/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/learning-agency-lab-automated-essay-scoring-2/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/learning-agency-lab-automated-essay-scoring-2/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> data/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import pandas as pd
import numpy as np
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from sklearn.svm import LinearSVC
from collections import Counter
from scipy.sparse import hstack



## === cell 1
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 2
train_df = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
)
test_df = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
)
sample_submission = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
)



## === cell 3
cList = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it had",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there had",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'alls": "you alls",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
}
c_re = re.compile("(%s)" % "|".join(cList.keys()))




## === cell 4
def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]

    return c_re.sub(replace, text)


def removeHTML(x):
    html = re.compile(r"<.*?>")
    return html.sub("", x)


def dataPreprocessing(series):
    series = series.str.lower()
    series = series.apply(removeHTML)
    series = series.apply(lambda s: re.sub("@\\w+", "", s))
    series = series.apply(lambda s: re.sub("'\\d+", "", s))
    series = series.apply(lambda s: re.sub("\\d+", "", s))
    series = series.apply(lambda s: re.sub("http\\w+", "", s))
    series = series.apply(lambda s: re.sub(r"\\s+", " ", s))
    series = series.apply(expandContractions)
    series = series.apply(lambda s: re.sub(r"\\.+", ".", s))
    series = series.apply(lambda s: re.sub(r",+", ",", s))
    series = series.apply(lambda s: re.sub("\n", "", s))
    series = series.apply(lambda s: re.sub("[^\\w\\s]", "", s))
    series = series.str.strip()
    return series




## === cell 5
X_text = dataPreprocessing(train_df["full_text"])
X_test_text = dataPreprocessing(test_df["full_text"])
y = train_df["score"]



## === cell 6
X_train_text, X_val_text, y_train, y_val = train_test_split(
    X_text, y, test_size=0.1, random_state=123, stratify=y
)



## === cell 7
vectorizer_word = TfidfVectorizer(
    stop_words="english",
    sublinear_tf=True,
    strip_accents="unicode",
    analyzer="word",
    token_pattern=r"\w{2,}",
    ngram_range=(1, 4),
    norm="l2",
    use_idf=True,
    smooth_idf=True,
    max_features=500000,
    min_df=1,
)

vectorizer_char = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=2,
    max_features=200000,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
    smooth_idf=True,
)

X_train_word = vectorizer_word.fit_transform(X_train_text)
X_train_char = vectorizer_char.fit_transform(X_train_text)
X_train_vec = hstack([X_train_word, X_train_char], format="csr")

X_val_word = vectorizer_word.transform(X_val_text)
X_val_char = vectorizer_char.transform(X_val_text)
X_val_vec = hstack([X_val_word, X_val_char], format="csr")



## === cell 8
clf = LinearSVC(
    C=30.0,  # stronger regularisation
    dual=False,
    random_state=123,
    max_iter=40000,
    class_weight="balanced",
)
clf.fit(X_train_vec, y_train)

clf_alt = LinearSVC(
    C=10.0,
    dual=False,
    random_state=124,
    max_iter=20000,
    class_weight="balanced",
)
clf_alt.fit(X_train_vec, y_train)

val_pred = clf.predict(X_val_vec)
val_pred_alt = clf_alt.predict(X_val_vec)
val_pred_ens = np.rint(0.7 * val_pred + 0.3 * val_pred_alt).astype(int)

mapping = {}
for pred_class in range(1, 7):
    mask = val_pred_ens == pred_class
    if mask.sum() > 0:
        most_common = Counter(y_val[mask]).most_common(1)[0][0]
        mapping[pred_class] = most_common
    else:
        mapping[pred_class] = pred_class

val_pred_mapped = np.vectorize(mapping.get)(val_pred_ens)

kappa_primary = cohen_kappa_score(y_val, val_pred, weights="quadratic")
kappa_ensemble = cohen_kappa_score(y_val, val_pred_ens, weights="quadratic")
kappa_mapped = cohen_kappa_score(y_val, val_pred_mapped, weights="quadratic")
print("Validation QWK (primary model):", kappa_primary)
print("Validation QWK (weighted ensemble):", kappa_ensemble)
print("Validation QWK (ensemble + mode mapping):", kappa_mapped)



## === cell 9
X_full_word = vectorizer_word.fit_transform(X_text)
X_full_char = vectorizer_char.fit_transform(X_text)
X_full_vec = hstack([X_full_word, X_full_char], format="csr")

clf_full = LinearSVC(
    C=30.0,
    dual=False,
    random_state=123,
    max_iter=40000,
    class_weight="balanced",
)
clf_full.fit(X_full_vec, y)

clf_full_alt = LinearSVC(
    C=10.0,
    dual=False,
    random_state=124,
    max_iter=20000,
    class_weight="balanced",
)
clf_full_alt.fit(X_full_vec, y)



## === cell 10
test_word = vectorizer_word.transform(X_test_text)
test_char = vectorizer_char.transform(X_test_text)
test_vec = hstack([test_word, test_char], format="csr")

pred_primary = clf_full.predict(test_vec)
pred_alt = clf_full_alt.predict(test_vec)

test_predictions = np.rint(0.7 * pred_primary + 0.3 * pred_alt).astype(int)

test_predictions = np.vectorize(mapping.get)(test_predictions)

test_predictions = np.clip(test_predictions, 1, 6)



## === cell 11
submission = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
)
submission["score"] = test_predictions
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
display(submission.head())
