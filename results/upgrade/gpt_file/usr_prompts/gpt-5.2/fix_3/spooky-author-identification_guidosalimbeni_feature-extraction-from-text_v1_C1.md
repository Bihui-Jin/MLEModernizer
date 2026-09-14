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

0.53379

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.52895) has done: 'Your score is worse than the target (0.49344 vs 0.44278; lower is better), so we should make small, safe changes that improve logloss without changing the overall approach (still: spaCy lemmatization + bag-of-words + MultinomialNB). The biggest issue is that `CountVectorizer()` with all defaults is usually suboptimal for this competition; adding word n-grams and mild frequency filtering typically improves logloss noticeably while keeping the same model. I also make the train/validation split deterministic (random_state) to stabilize local logloss comparisons, and I fix the missing cell number (cell 24) so the script is a clean, fully runnable sequence that always produces `submission_v2.csv`. These changes are directly aimed at improving the probability estimates used by logloss while keeping the same core pipeline.'
- What this solution (achieved 0.53379) has done: 'Your current score (0.52895, lower is better) is worse than the target (0.44278), so we should make a small, safe improvement that keeps the same pipeline (spaCy lemmatization → bag-of-words → MultinomialNB) but typically reduces logloss. The least invasive win here is to add simple token-level normalization inside `cleanup_text` (remove leftover non-alphabetic tokens and very short tokens that add noise) while keeping the same vectorizer/model; this often improves Naive Bayes calibration on this dataset. I’m also keeping the deterministic split and the same submission schema, and I’m only fixing the missing cell number so the notebook runs cleanly end-to-end and always writes `submission_v2.csv`. These changes are directly aimed at improving probabilistic predictions (logloss) without changing the overall approach.'

# 9. Code solution

## === cell 0
import os

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
    """Multi class version of Logarithmic Loss metric.
    :param actual: Array containing the actual target classes
    :param predicted: Matrix with class predictions, one probability per class
    """
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
        tokens = [tok for tok in tokens if tok.isalpha() and len(tok) > 2]

        tokens = " ".join(tokens)
        texts.append(tokens)
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
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)



## === cell 16
X_train_matrix = vect.fit_transform(X_train)



## === cell 17
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
clf.fit(X_train_matrix, y_train)
print(clf.score(X_train_matrix, y_train))
X_test_matrix = vect.transform(X_test)
print(clf.score(X_test_matrix, y_test))



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
print("Original training data shape: ", test["text"].shape)
test["test_cleaned"] = cleanup_text(test["text"], logging=True)
print("Cleaned up training data shape: ", test["test_cleaned"].shape)



## === cell 23
test_matrix = vect.transform(test["test_cleaned"])
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
print("Wrote submission_v2.csv with shape:", result.shape)
