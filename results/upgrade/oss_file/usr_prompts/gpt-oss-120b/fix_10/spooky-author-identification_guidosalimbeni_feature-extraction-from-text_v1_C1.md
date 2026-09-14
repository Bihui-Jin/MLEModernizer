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

3.8

# 3. Installed packages

No external packages required in the script and installed.

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

0.44278

# 6. Current score

0.51983

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.67753) has done: 'I keep the overall Naive‑Bayes + CountVectorizer pipeline but tighten the preprocessing and model hyper‑parameters so the validation log‑loss moves closer to the target (lower is better).  
Key tweaks: set a deterministic random seed, give `train_test_split` a fixed `random_state`, use a richer bag‑of‑words (`ngram_range=(1,2)` and a caps on features), and apply a slightly stronger smoothing (`alpha=0.1`). These changes stay within the original logic but should improve calibration and reduce the log‑loss.'
- What this solution (achieved 0.56972) has done: 'I increase the feature space size and switch from MultinomialNB to ComplementNB, which usually yields better calibrated probabilities for multiclass text classification. These changes stay within the original Naïve Bayes + bag‑of‑words pipeline, keep the same preprocessing, and are aimed at lowering the log‑loss toward the target value.'
- What this solution (achieved 0.5392) has done: 'I add a TF‑IDF transformation after the CountVectorizer while keeping the same Naïve Bayes model. This small change often improves probability calibration and lowers log‑loss, moving the score closer to the target without altering the core pipeline.'
- What this solution (achieved 0.53772) has done: 'I tighten the preprocessing and model hyper‑parameters slightly to improve probability calibration and lower the log‑loss toward the target. In the vectorizer I drop very rare tokens (`min_df=2`) and in the TF‑IDF step I enable sub‑linear term frequency scaling. I also reduce the ComplementNB smoothing parameter from 0.1 to 0.05. These changes keep the overall Naïve‑Bayes + bag‑of‑words pipeline intact while providing a modest boost in performance.'
- What this solution (achieved 0.5029) has done: 'I slightly adjust the text‑vectorizer and the ComplementNB smoothing parameter – both are minimal hyper‑parameter tweaks that keep the original Naïve Bayes + bag‑of‑words pipeline unchanged. Expanding the n‑gram range and allowing a few more features lets the model capture richer patterns, while lowering α from 0.05 to 0.01 usually yields better calibrated probabilities, which should reduce the multi‑class log‑loss toward the target.'
- What this solution (achieved 0.55346) has done: 'I slightly adjust the text vectorizer and ComplementNB smoothing to improve probability calibration and lower the log‑loss toward the target (0.44278). Specifically, the `CountVectorizer` use a more conservative n‑gram range (1‑2), drop very rare tokens (`min_df=2`), and reduce the maximum feature count. The `ComplementNB` classifier use a slightly larger smoothing parameter (`alpha=0.1`). These changes stay within the original Naïve Bayes + bag‑of‑words pipeline and are minimal yet expected to reduce the validation log‑loss.'
- What this solution (achieved 0.50614) has done: 'I keep the original Naïve Bayes + CountVectorizer + TF‑IDF pipeline but make two small hyper‑parameter tweaks that usually improve probability calibration: (1) increase the vocabulary size (`max_features`) to capture more informative tokens, and (2) lower the ComplementNB smoothing parameter (`alpha`) to 0.05. These adjustments stay within the core logic and are expected to lower the multi‑class log‑loss, moving the score closer to the target 0.44278. No other parts of the script are changed, and the final CSV is still written with the required column order.'
- What this solution (achieved 0.5318) has done: 'I slightly expand the vocabulary and raise the n‑gram range in the CountVectorizer (to capture more informative patterns) and reduce the ComplementNB smoothing (α = 0.01) so the model’s probability estimates become better calibrated, which should lower the multi‑class log‑loss toward the target value while keeping the original pipeline unchanged.'
- What this solution (achieved 0.51983) has done: 'I tighten the text vectorizer and slightly adjust the ComplementNB smoothing to improve probability calibration, which should lower the validation log‑loss and move the score closer to the target (0.44278). The core pipeline (cleanup, CountVectorizer → TFIDF → ComplementNB) remains unchanged, and the script still writes a correctly‑formatted `submission_v2.csv`.'

# 9. Code solution

## === cell 0
import os, numpy as np, random, pandas as pd, spacy, seaborn as sns, string

np.random.seed(42)
random.seed(42)
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import numpy as np
import pandas as pd
import spacy
import seaborn as sns
import string




## === cell 2
def multiclass_logloss(actual, predicted, eps=1e-15):
    """Multi class version of Logarithmic Loss metric."""
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2
    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota




## === cell 3
nlp = spacy.load("en_core_web_sm")



## === cell 4
data = pd.read_csv("/kaggle/input/spooky-author-identification/train.csv")
data.head()



## === cell 5
doc = nlp(data["text"][0])
for token in doc:
    print(
        token.text, token.pos, token.pos_, token.dep_
    )  # part of speach and syntax dependency



## === cell 6
for token in doc:
    print(token.text, token.pos_, token.lemma_)  # part of speach and syntax dependency



## === cell 7
sns.barplot(
    x=["Edgar Allen Poe", "Mary Wollstonecraft Shelley", "H.P. Lovecraft"],
    y=data["author"].value_counts(),
)



## === cell 8
data["author_num"] = data["author"].map({"EAP": 0, "HPL": 1, "MWS": 2})
data.head()



## === cell 9
from nltk.corpus import stopwords

try:
    stopwords = stopwords.words("english")
except LookupError:
    import nltk

    nltk.download("stopwords")
    stopwords = stopwords.words("english")
print(stopwords)



## === cell 10
punctuations = string.punctuation


def cleanup_text(docs, logging=False):
    texts = []
    counter = 1
    for doc in docs:
        if counter % 1000 == 0 and logging:
            print("Processed %d out of %d documents." % (counter, len(docs)))
        counter += 1
        doc = nlp(doc, disable=["parser", "ner"])
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != "-PRON-"]
        tokens = [
            tok for tok in tokens if tok not in stopwords and tok not in punctuations
        ]
        texts.append(" ".join(tokens))
    return pd.Series(texts)




## === cell 11
print("Original training data shape: ", data["text"].shape)
train_cleaned = cleanup_text(data["text"], logging=True)
print("Cleaned up training data shape: ", train_cleaned.shape)



## === cell 12
data["train_cleaned"] = train_cleaned
data.head()



## === cell 13
X = data["train_cleaned"]
y = data["author_num"]



## === cell 14
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)



## === cell 15
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

vect = CountVectorizer(ngram_range=(1, 2), max_features=400000, min_df=1)
tfidf = TfidfTransformer(sublinear_tf=True)



## === cell 16
X_train_counts = vect.fit_transform(X_train)
X_train_matrix = tfidf.fit_transform(X_train_counts)



## === cell 17
from sklearn.naive_bayes import ComplementNB

clf = ComplementNB(alpha=0.005)
clf.fit(X_train_matrix, y_train)
print("Train accuracy:", clf.score(X_train_matrix, y_train))

X_test_counts = vect.transform(X_test)
X_test_matrix = tfidf.transform(X_test_counts)
print("Validation accuracy:", clf.score(X_test_matrix, y_test))



## === cell 18
predicted_result = clf.predict(X_test_matrix)
from sklearn.metrics import classification_report

print(classification_report(y_test, predicted_result))



## === cell 19
predictions = clf.predict_proba(X_test_matrix)
print("logloss: %0.3f " % multiclass_logloss(y_test, predictions))



## === cell 20
sample = pd.read_csv("/kaggle/input/spooky-author-identification/sample_submission.csv")
sample.head()



## === cell 21
test = pd.read_csv("/kaggle/input/spooky-author-identification/test.csv")



## === cell 22
print("Original test data shape: ", test["text"].shape)
test["test_cleaned"] = cleanup_text(test["text"], logging=True)
print("Cleaned test data shape: ", test["test_cleaned"].shape)



## === cell 23
test_counts = vect.transform(test["test_cleaned"])
test_matrix = tfidf.transform(test_counts)
predicted_result = clf.predict_proba(test_matrix)



## === cell 24
result = pd.DataFrame()
result["id"] = test["id"]
result["EAP"] = predicted_result[:, 0]
result["HPL"] = predicted_result[:, 1]
result["MWS"] = predicted_result[:, 2]
result.head()



## === cell 25
result.to_csv("submission_v2.csv", index=False)
