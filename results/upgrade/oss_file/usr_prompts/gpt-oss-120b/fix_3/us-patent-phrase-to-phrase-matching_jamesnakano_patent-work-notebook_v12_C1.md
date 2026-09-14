# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given pairs of phrases (an `anchor` and a `target` phrase), build a model to rate how similar they are.  

## Metric
Pearson correlation coefficient.

## Submission Format
For each `id` (representing a pair of phrases) in the test set, you must predict the similarity `score`. The file should contain a header and have the following format:

```
id,score
4112d61851461f60,0
09e418c93a776564,0.25
36baf228038e314b,1
etc.

```

## Dataset
The scores are in the 0-1 range with increments of 0.25 with the following meanings:

- **1.0** - Very close match. This is typically an exact match except possibly for differences in conjugation, quantity (e.g. singular vs. plural), and addition or removal of stopwords (e.g. "the", "and", "or").
- **0.75** - Close synonym, e.g. "mobile phone" vs. "cellphone". This also includes abbreviations, e.g. "TCP" -> "transmission control protocol".
- **0.5** - Synonyms which don't have the same meaning (same function, same properties). This includes broad-narrow (hyponym) and narrow-broad (hypernym) matches.
- **0.25** - Somewhat related, e.g. the two phrases are in the same high level domain but are not synonyms. This also includes antonyms.
- **0.0** - Unrelated.

Files
-----

- **train.csv** - the training set, containing phrases, contexts, and their similarity scores
- **test.csv** - the test set set, identical in structure to the training set but without the score
- **sample_submission.csv** - a sample submission file in the correct format

Columns
-------

- `id` - a unique identifier for a pair of phrases
- `anchor` - the first phrase
- `target` - the second phrase
- `context` - the [CPC classification (version 2021.05)](https://en.wikipedia.org/wiki/Cooperative_Patent_Classification), which indicates the subject within which the similarity is to be scored
- `score` - the similarity. This is sourced from a combination of one or more manual expert ratings.

# 2. Python version

3.10

# 3. Installed packages

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
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        input/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        working/
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
```

-> data/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> data/us-patent-phrase-to-phrase-matching/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/us-patent-phrase-to-phrase-matching/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/us-patent-phrase-to-phrase-matching/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> input/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import LancasterStemmer
from sklearn.ensemble import RandomForestClassifier
import pickle



## === cell 1
df = pd.read_csv("../input/us-patent-phrase-to-phrase-matching/train.csv")
test = pd.read_csv("../input/us-patent-phrase-to-phrase-matching/test.csv")
sub = pd.read_csv("../input/us-patent-phrase-to-phrase-matching/sample_submission.csv")



## === cell 2
df["score"] = df["score"].apply(
    lambda x: (
        0
        if x == 0.0
        else (
            1
            if x == 0.25
            else 2 if x == 0.5 else 3 if x == 0.75 else 4 if x == 1 else x
        )
    )
)
stop_words = stopwords.words("english")
ls = LancasterStemmer()
df["letter"] = df["context"].apply(lambda x: x[0])
df["anchor_token"] = df["anchor"].apply(lambda x: word_tokenize(x))
df["anchor_nostop"] = df["anchor_token"].apply(
    lambda x: [word for word in x if word not in stop_words]
)
df["anchor_stem"] = df["anchor_nostop"].apply(lambda x: [ls.stem(word) for word in x])
df["new_anchor"] = df["anchor_stem"].apply(
    lambda x: " ".join([str(word) for word in x])
)
df["target_token"] = df["target"].apply(lambda x: word_tokenize(x))
df["target_nostop"] = df["target_token"].apply(
    lambda x: [word for word in x if word not in stop_words]
)
df["target_stem"] = df["target_nostop"].apply(lambda x: [ls.stem(word) for word in x])
df["new_target"] = df["target_stem"].apply(lambda x: " ".join([word for word in x]))
df["final"] = df["new_anchor"] + " " + df["new_target"]



## === cell 3
stop_words = stopwords.words("english")
ls = LancasterStemmer()
test["letter"] = test["context"].apply(lambda x: x[0])
test["anchor_token"] = test["anchor"].apply(lambda x: word_tokenize(x))
test["anchor_nostop"] = test["anchor_token"].apply(
    lambda x: [word for word in x if word not in stop_words]
)
test["anchor_stem"] = test["anchor_nostop"].apply(
    lambda x: [ls.stem(word) for word in x]
)
test["new_anchor"] = test["anchor_stem"].apply(
    lambda x: " ".join([str(word) for word in x])
)
test["target_token"] = test["target"].apply(lambda x: word_tokenize(x))
test["target_nostop"] = test["target_token"].apply(
    lambda x: [word for word in x if word not in stop_words]
)
test["target_stem"] = test["target_nostop"].apply(
    lambda x: [ls.stem(word) for word in x]
)
test["new_target"] = test["target_stem"].apply(lambda x: " ".join([word for word in x]))
test["final"] = test["new_anchor"] + " " + test["new_target"]



## === cell 4
vectorizer = CountVectorizer()
vectortrain = vectorizer.fit(df["final"])
matrixtrain = vectorizer.transform(df["final"])
trainvect = pd.DataFrame(
    matrixtrain.toarray(), columns=vectortrain.get_feature_names_out()
)
trainvect["score"] = df.reset_index().score
trainx, testx, trainy, testy = train_test_split(
    trainvect.drop("score", axis=1), trainvect["score"], test_size=0.2, random_state=777
)
rfc = RandomForestClassifier(n_estimators=100)



## === cell 5
pd.options.mode.chained_assignment = None
matrixtest = vectorizer.transform(test["final"])
dftest = pd.DataFrame(matrixtest.toarray(), columns=vectortrain.get_feature_names_out())
rfcfit = rfc.fit(trainvect.drop("score", axis=1), trainvect["score"])

pred_int = rfcfit.predict(dftest).astype(float)

map_dict = {0: 0.0, 1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0}
pred = np.vectorize(map_dict.get)(pred_int)

np.random.seed(42)
noise = np.random.normal(
    loc=0.0, scale=0.30, size=pred.shape
)  # larger stddev for stronger perturbation
pred = np.clip(pred + noise, 0.0, 1.0)

sub["score"] = pred
sub.to_csv("submission.csv", index=False)



## === cell 6
sub
