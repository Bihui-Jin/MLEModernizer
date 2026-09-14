# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

gensim==4.4.0
geopandas==0.14.4
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
sklearn-pandas==2.2.0

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

1.05968

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import nltk
from gensim.models import Word2Vec

from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

alpha_tokenizer = RegexpTokenizer("[A-Za-z]\\w+")
lemmatizer = WordNetLemmatizer()
stop = stopwords.words("english")



## === cell 1
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")



## === cell 2
train_tokens = [
    [
        lemmatizer.lemmatize(word.lower())
        for word in alpha_tokenizer.tokenize(sent)
        if word.lower() not in stop
    ]
    for sent in train.text.values
]



## === cell 3
NUM_FEATURES = 100
model = Word2Vec(
    train_tokens, vector_size=NUM_FEATURES, min_count=3, window=6, sg=0, workers=4
)



## === cell 4
vocab_size = len(model.wv.key_to_index)




## === cell 5
def get_feature_vec(tokens, num_features, wv):
    """Average of word vectors for the given token list."""
    feature_vec = np.zeros(num_features, dtype="float32")
    missed = 0
    for word in tokens:
        if word in wv:
            feature_vec += wv[word]
        else:
            missed += 1
    valid_cnt = len(tokens) - missed
    if valid_cnt == 0:
        return np.zeros(num_features, dtype="float32")
    return feature_vec / valid_cnt




## === cell 6
train_vectors = [
    get_feature_vec(
        [
            lemmatizer.lemmatize(word.lower())
            for word in alpha_tokenizer.tokenize(sent)
            if word.lower() not in stop
        ],
        NUM_FEATURES,
        model.wv,
    )
    for sent in train.text.values
]
X_train = np.vstack(train_vectors)



## === cell 7
train["author"] = train["author"].map({"EAP": 0, "HPL": 1, "MWS": 2})
y_train = train.author.values



## === cell 8
from sklearn.linear_model import LogisticRegression

estimator = LogisticRegression(
    C=1, max_iter=200, multi_class="multinomial", solver="lbfgs"
)
estimator.fit(X_train, y_train)



## === cell 9
test_vectors = [
    get_feature_vec(
        [
            lemmatiser.lemmatize(word.lower())
            for word in alpha_tokenizer.tokenize(sent)
            if word.lower() not in stop
        ],
        NUM_FEATURES,
        model.wv,
    )
    for sent in test.text.values
]
X_test = np.vstack(test_vectors)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3037679385.py in <cell line: 0>()
----> 1 test_vectors = [
      2     get_feature_vec(
      3         [
      4             lemmatiser.lemmatize(word.lower())
      5             for word in alpha_tokenizer.tokenize(sent)

/tmp/ipykernel_11/3037679385.py in <listcomp>(.0)
      1 test_vectors = [
      2     get_feature_vec(
----> 3         [
      4             lemmatiser.lemmatize(word.lower())
      5             for word in alpha_tokenizer.tokenize(sent)

/tmp/ipykernel_11/3037679385.py in <listcomp>(.0)
      2     get_feature_vec(
      3         [
----> 4             lemmatiser.lemmatize(word.lower())
      5             for word in alpha_tokenizer.tokenize(sent)
      6             if word.lower() not in stop

NameError: name 'lemmatiser' is not defined

## === cell 10
probs = estimator.predict_proba(X_test)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/673271032.py in <cell line: 0>()
----> 1 probs = estimator.predict_proba(X_test)
      2 

NameError: name 'X_test' is not defined

## === cell 11
author_probs = pd.DataFrame(probs, columns=["EAP", "HPL", "MWS"])
submission = pd.concat([test["id"].reset_index(drop=True), author_probs], axis=1)
submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2268984226.py in <cell line: 0>()
----> 1 author_probs = pd.DataFrame(probs, columns=["EAP", "HPL", "MWS"])
      2 submission = pd.concat([test["id"].reset_index(drop=True), author_probs], axis=1)
      3 submission.to_csv("submission.csv", index=False)

NameError: name 'probs' is not defined
