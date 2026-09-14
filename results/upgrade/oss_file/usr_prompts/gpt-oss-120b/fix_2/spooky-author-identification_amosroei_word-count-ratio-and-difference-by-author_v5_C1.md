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

3.7

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
xgboost==2.0.3

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

0.90334

# 6. Current score

0.56399

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 0.56399) has done: 'The changes fix the deprecated `get_feature_names` calls, ensure the feature dictionary is built correctly, and remove the invalid XGBoost parameter so the model can train on the engineered features and produce a valid submission CSV.'

# 9. Code solution

## === cell 0
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from itertools import islice
import textwrap
from sklearn.model_selection import train_test_split

wrapper = textwrap.TextWrapper(initial_indent="", width=70, subsequent_indent=" " * 3)

import nltk

nltk.download("wordnet")
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
nltk.download("vader_lexicon")


## === cell 1
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

text_column = "text"
label = "author"


## === cell 2
train_df.head()


## === cell 3
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD

import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from nltk.sentiment.vader import SentimentIntensityAnalyzer

english_stopwords = set(stopwords.words("english"))


## === cell 4
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

porter_stemmer = PorterStemmer()
lemm = WordNetLemmatizer()


class LemmaCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(LemmaCountVectorizer, self).build_analyzer()
        return lambda doc: (
            porter_stemmer.stem(lemm.lemmatize(w)) for w in analyzer(doc)
        )


eap_text = list(train_df[train_df["author"] == "EAP"][text_column].values)
hpl_text = list(train_df[train_df["author"] == "HPL"][text_column].values)
mws_text = list(train_df[train_df["author"] == "MWS"][text_column].values)

author_text_dict = dict(zip([0, 1, 2], [eap_text, hpl_text, mws_text]))

full_text = eap_text + mws_text + hpl_text

full_tf_vectorizer = LemmaCountVectorizer(
    max_df=0.95, min_df=2, stop_words="english", decode_error="ignore"
)
full_tf = full_tf_vectorizer.fit_transform(full_text)
full_feature_names = full_tf_vectorizer.get_feature_names_out()

author_word_freq_df = pd.DataFrame(0.0, index=[0, 1, 2], columns=full_feature_names)

author_wordcount_dict = {}

for author, text in author_text_dict.items():
    tf_vectorizer = LemmaCountVectorizer(
        max_df=0.95, min_df=2, stop_words="english", decode_error="ignore"
    )
    tf = tf_vectorizer.fit_transform(text)
    feature_names = tf_vectorizer.get_feature_names_out()
    count_vec = np.asarray(tf.sum(axis=0)).ravel()
    zipped = list(zip(feature_names, count_vec))
    author_wordcount_dict[author] = zipped


## === cell 5
for author, zipped in author_wordcount_dict.items():
    for word, count in zipped:
        author_word_freq_df[word.lower()][author] = count

transposed_freq_df = author_word_freq_df.T

transposed_freq_df["0_count"] = (
    transposed_freq_df[0] - transposed_freq_df[1] - transposed_freq_df[2]
)
transposed_freq_df["1_count"] = (
    transposed_freq_df[1] - transposed_freq_df[0] - transposed_freq_df[2]
)
transposed_freq_df["2_count"] = (
    transposed_freq_df[2] - transposed_freq_df[0] - transposed_freq_df[1]
)

epsilon = 1
transposed_freq_df["0_ratio"] = (transposed_freq_df[0] + epsilon) / (
    transposed_freq_df[1] + transposed_freq_df[2] + epsilon
)
transposed_freq_df["1_ratio"] = (transposed_freq_df[1] + epsilon) / (
    transposed_freq_df[0] + transposed_freq_df[2] + epsilon
)
transposed_freq_df["2_ratio"] = (transposed_freq_df[2] + epsilon) / (
    transposed_freq_df[0] + transposed_freq_df[1] + epsilon
)

transposed_freq_df.sort_values(by="0_ratio", ascending=False)




## === cell 6
def calc_count_score(text, author):
    word_list = word_tokenize(text)
    if len(word_list) == 0:
        return 0.0
    score = 0.0
    for word in word_list:
        lemm_word = porter_stemmer.stem(lemm.lemmatize(word))
        if lemm_word in transposed_freq_df.index:
            score += transposed_freq_df[str(author) + "_count"][lemm_word]
    return score / len(word_list)


def calc_ratio_score(text, author):
    word_list = word_tokenize(text)
    if len(word_list) == 0:
        return 1.0
    score = 1.0
    for word in word_list:
        lemm_word = porter_stemmer.stem(lemm.lemmatize(word))
        if lemm_word in transposed_freq_df.index:
            score *= transposed_freq_df[str(author) + "_ratio"][lemm_word]
    return score / len(word_list)


train_df["eap_freq_count_score"] = train_df[text_column].apply(
    lambda row: calc_count_score(row, 0)
)
train_df["hpl_freq_count_score"] = train_df[text_column].apply(
    lambda row: calc_count_score(row, 1)
)
train_df["mws_freq_count_score"] = train_df[text_column].apply(
    lambda row: calc_count_score(row, 2)
)

train_df["eap_freq_ratio_score"] = train_df[text_column].apply(
    lambda row: calc_ratio_score(row, 0)
)
train_df["hpl_freq_ratio_score"] = train_df[text_column].apply(
    lambda row: calc_ratio_score(row, 1)
)
train_df["mws_freq_ratio_score"] = train_df[text_column].apply(
    lambda row: calc_ratio_score(row, 2)
)

test_df["eap_freq_count_score"] = test_df[text_column].apply(
    lambda row: calc_count_score(row, 0)
)
test_df["hpl_freq_count_score"] = test_df[text_column].apply(
    lambda row: calc_count_score(row, 1)
)
test_df["mws_freq_count_score"] = test_df[text_column].apply(
    lambda row: calc_count_score(row, 2)
)

test_df["eap_freq_ratio_score"] = test_df[text_column].apply(
    lambda row: calc_ratio_score(row, 0)
)
test_df["hpl_freq_ratio_score"] = test_df[text_column].apply(
    lambda row: calc_ratio_score(row, 1)
)
test_df["mws_freq_ratio_score"] = test_df[text_column].apply(
    lambda row: calc_ratio_score(row, 2)
)


## === cell 7
test_id = test_df["id"].values

author_mapping_dict = {"EAP": 0, "HPL": 1, "MWS": 2}
cols_to_drop = ["id", "text"]
X_train = train_df.drop(cols_to_drop + ["author"], axis=1)
X_test = test_df.drop(cols_to_drop, axis=1)

y_train = train_df["author"].map(author_mapping_dict)


## === cell 8
xgb_clf = xgb.XGBClassifier(
    objective="multi:softprob",
    colsample_bytree=0.3,
    learning_rate=0.1,
    max_depth=3,
    alpha=10,
    n_estimators=100,
    use_label_encoder=False,
    eval_metric="mlogloss",
    verbosity=0,
)
xgb_clf.fit(X_train, y_train)


## === cell 9
y_pred = xgb_clf.predict_proba(X_test)


## === cell 10
out_df = pd.DataFrame(y_pred, columns=["EAP", "HPL", "MWS"])
out_df.insert(0, "id", test_id)
out_df.to_csv("sub_fe.csv", index=False)
