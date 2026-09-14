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
Given a dataset of comments from Wikipedia's talk page edits, predict the probability of each comment being toxic.

## Metric
Mean column-wise ROC AUC; the average of the individual AUCs of each predicted column.

## Submission Format
For each `id` in the test set, you must predict a probability for each of the six possible types of comment toxicity (toxic, severe_toxic, obscene, threat, insult, identity_hate). The columns must be in the same order as shown below. The file should contain a header and have the following format:

```
id,toxic,severe_toxic,obscene,threat,insult,identity_hate
00001cee341fdb12,0.5,0.5,0.5,0.5,0.5,0.5
0000247867823ef7,0.5,0.5,0.5,0.5,0.5,0.5
etc.
```

## Dataset 
- **train.csv** - the training set, contains comments with their binary labels
- **test.csv** - the test set, you must predict the toxicity probabilities for these comments.
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.12

# 3. Installed packages

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        input/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        working/
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
```

-> data/jigsaw-toxic-comment-classification-challenge/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/jigsaw-toxic-comment-classification-challenge/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/jigsaw-toxic-comment-classification-challenge/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> (stopped after 10 files for performance)

# 5. Target score

0.94991

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import re
import shutil
from collections import Counter

import numpy as np
import pandas as pd
from IPython.display import display
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain
from tqdm import tqdm

DATA_DIR = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/"
OUTPUT_DIR = "/kaggle/working/"


## === cell 1
def unpack_zipfile(filename):
    """Unpacks zip-file by name from DATA_DIR to OUTPUT_DIR."""
    try:
        shutil.unpack_archive(
            filename=DATA_DIR + filename,
            extract_dir=OUTPUT_DIR,
            format="zip",
        )
    except Exception as e:
        print(e)
    else:
        print(f"Archive file '{filename}' has been unpacked successfully.")


## === cell 2
unpack_zipfile(filename="train.csv.zip")
unpack_zipfile(filename="test.csv.zip")
unpack_zipfile(filename="test_labels.csv.zip")


## === cell 3
train_df = pd.read_csv(OUTPUT_DIR + "train.csv")
test_df = pd.read_csv(OUTPUT_DIR + "test.csv")


## === cell 4
train_df.head()


## === cell 5
test_df.head()


## === cell 6
for df in (train_df, test_df):
    df["comment_text_preprocessed"] = df["comment_text"].str.lower()


## === cell 7
cols = ["comment_text", "comment_text_preprocessed"]
display(train_df[cols].sample(5))
display(test_df[cols].sample(5))


## === cell 8
eng_stopwords = set(nltk_stopwords.words('english'))
eng_stopwords.update(["i'm", "that's", "can't"])
eng_stopwords


## === cell 9
def clear_stopwords(comment_text, stopwords=eng_stopwords):
    """Removes stopwords from the commentary text."""
    comment_text_cleared = [word for word in str(comment_text).split() 
                              if word not in stopwords]
    
    return " ".join(comment_text_cleared)


## === cell 10
train_text_2 = train_df["comment_text_preprocessed"].iloc[2]

print("Inp:\n\n{}\n".format(train_text_2))
print("Out:\n\n{}".format(clear_stopwords(train_text_2)))


## === cell 11
for df in (train_df, test_df):
    df["comment_text_preprocessed"] = df["comment_text_preprocessed"] \
    .apply(
        lambda comment_text: clear_stopwords(comment_text)
    )


## === cell 12
display(train_df[cols].sample(5))
display(test_df[cols].sample(5))


## === cell 13
word_counter = Counter()
for comment_text in train_df["comment_text_preprocessed"].values:
    for word in comment_text.split():
        word_counter[word] += 1

word_counter.most_common(10)


## === cell 14
freq_words = set([word for (word, word_count) in word_counter.most_common(10)])
freq_words


## === cell 15
def clear_freqwords(comment_text, freqwords=freq_words):
    """Removes top-10 frequent words."""
    
    comment_text_cleared = [word for word in str(comment_text).split() 
                              if word not in freq_words]
    
    return " ".join(comment_text_cleared)


## === cell 16
for df in (train_df, test_df):
    df["comment_text_preprocessed"] = df["comment_text_preprocessed"] \
    .apply(
        lambda comment_text: clear_freqwords(comment_text)
    )


## === cell 17
display(train_df[cols].sample(5))
display(test_df[cols].sample(5))


## === cell 18
rare_words_num = 10
rare_words = set([word for (word, word_count) 
                  in word_counter.most_common()[:-rare_words_num-1:-1]])
rare_words


## === cell 19
def clear_rarewords(comment_text, rarewords=rare_words):
    """Removes top-10 rarest words."""
    
    comment_text_cleared = [word for word in str(comment_text).split() 
                              if word not in rare_words]
    
    return " ".join(comment_text_cleared)


## === cell 20
for df in (train_df, test_df):
    df["comment_text_preprocessed"] = df["comment_text_preprocessed"] \
    .apply(
        lambda comment_text: clear_rarewords(comment_text)
    )


## === cell 21
display(train_df[cols].sample(5))
display(test_df[cols].sample(5))


## === cell 22
def clear_urls(comment_text):
    """Clears the comment text from URLs."""
    
    url_regex_pattern = re.compile(r'https?://\S+|www\.\S+')
    
    return url_regex_pattern.sub(r"", comment_text)


## === cell 23
train_text_900 = train_df["comment_text_preprocessed"].iloc[-900]

print("Inp:\n\n{}\n".format(train_text_900))
print("Out:\n\n{}".format(clear_urls(train_text_900)))


## === cell 24
for df in (train_df, test_df):
    df["comment_text_preprocessed"] = df["comment_text_preprocessed"] \
    .apply(
        lambda comment_text: clear_urls(comment_text)
    )


## === cell 25
display(train_df[cols].sample(5))
display(test_df[cols].sample(5))


## === cell 26
regex = re.compile(r"[a-zA-Z]+")

def leave_words_only(comment_text, regex=regex):
    """Removes non-word inclusions."""
    
    return " ".join(regex.findall(comment_text))


## === cell 27
train_text = train_df["comment_text_preprocessed"].iloc[-1]

print("Inp:\n\n{}\n".format(train_text))
print("Out:\n\n{}".format(leave_words_only(train_text)))


## === cell 28
for df in (train_df, test_df):
    df["comment_text_preprocessed"] = df["comment_text_preprocessed"] \
    .apply(
        lambda comment_text: leave_words_only(comment_text)
    )


## === cell 29
display(train_df[cols].sample(5))
display(test_df[cols].sample(5))


## === cell 30
other_eng_stopwords = [word for word in eng_stopwords if "'" in word]
other_eng_stopwords


## === cell 31
other_eng_stopwords = [word.replace("'", "") for word in other_eng_stopwords]
other_eng_stopwords


## === cell 32
word_counter = Counter()
for comment_text in train_df["comment_text_preprocessed"].values:
    for word in comment_text.split():
        word_counter[word] += 1

word_counter.most_common(100)


## === cell 33
eng_stopwords.update(
    [
        "utc", "eg", 
        "jpg", "didnt",
        "th", "oh", 
        "im", "cant", 
        "wp", "hi",
    ]
)
eng_stopwords.update(other_eng_stopwords)
eng_stopwords


## === cell 34
for df in (train_df, test_df):
    df["comment_text_preprocessed"] = df["comment_text_preprocessed"] \
    .apply(
        lambda comment_text: clear_stopwords(
            comment_text, stopwords=eng_stopwords,
        )
    )


## === cell 35
for df in (train_df, test_df):
    df["comment_text_preprocessed"] = df["comment_text_preprocessed"] \
    .apply(
        lambda comment_text: clear_freqwords(comment_text)
    )


## === cell 36
display(train_df[cols].sample(5))
display(test_df[cols].sample(5))


## === cell 37
target_cols = train_df.columns[2:-1]
target_train = train_df[target_cols].values
target_train[:5]


## === cell 38
corpus_train = train_df["comment_text_preprocessed"].values.astype("U")
corpus_train[:5]


## === cell 39
corpus_test = test_df["comment_text_preprocessed"].values.astype("U")
corpus_test[:5]


## === cell 40
vectorizer = TfidfVectorizer(
    max_features=1700,
    min_df=0.0011,
    max_df=0.35,
    norm="l2",
)


## === cell 41
features_train = vectorizer.fit_transform(corpus_train)
features_train.shape


## === cell 42
features_test = vectorizer.transform(corpus_test)
features_test.shape


## === cell 43
base_estimator = LogisticRegression(
    class_weight="balanced",
    max_iter=10000,
    multi_class="multinomial",
    C=0.009,
    penalty="l2",
    n_jobs=-1,
)


## === cell 44
chains = [
    ClassifierChain(
        base_estimator=base_estimator,
        order="random", 
        random_state=i,
    ) for i in range(10)
]

for i in tqdm(range(len(chains))):
    chains[i].fit(features_train, target_train)
print()


## === cell 45
predictions = np.array([chain.predict_proba(features_test) 
                          for chain in chains])
proba_predictions_test = predictions.mean(axis=0)
proba_predictions_test


## === cell 46
submission = pd.DataFrame(
    proba_predictions_test, 
    columns=target_cols,
    index=test_df.id
).reset_index()

submission.head()


## === cell 47
submission.info()


## === cell 48
submission.to_csv('submission.csv', index=False)

print("The submission has been successfully saved.")
