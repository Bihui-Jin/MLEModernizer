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

cudf-polars-cu12==25.6.0
geopandas==0.14.4
lightgbm==4.6.0
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
polars==1.25.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tqdm==4.67.1

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
import os
import gc
import ctypes
import random
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import pickle
from IPython.display import display

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import lightgbm as lgb
from lightgbm import early_stopping




## === cell 1
class CFG:
    SEED = 2024
    VER = 1
    LOAD_MODELS_FROM = None
    LOAD_FEATURES_FROM = None
    BASE_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/"


Clean = True


def clean_memory():
    if Clean:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        gc.collect()


clean_memory()




## === cell 2
def seed_everything():  # To produce similar result in each run
    random.seed(CFG.SEED)
    np.random.seed(CFG.SEED)
    os.environ["PYTHONHASHSEED"] = str(CFG.SEED)


seed_everything()




## === cell 3
df_train = pd.read_csv(CFG.BASE_PATH + "train.csv")
df_train = df_train.sort_values(by="essay_id")
print("Shape of Train: ", df_train.shape)
display(df_train.head())




## === cell 4
plt.figure(figsize=(12, 6))
sns.countplot(x=df_train["score"])
plt.title("Distribution of Score")
plt.xlabel("Score of Essay")
plt.ylabel("Frequency")
plt.show()




## === cell 5
df_test = pd.read_csv(CFG.BASE_PATH + "test.csv")
df_test = df_test.sort_values(by="essay_id")
print("Shape of Test: ", df_test.shape)
display(df_test.head())




## === cell 6
train = pl.from_pandas(df_train).with_columns(
    pl.col("full_text").str.split(by="\n\n").alias("paragraph")
)
test = pl.from_pandas(df_test).with_columns(
    pl.col("full_text").str.split(by="\n\n").alias("paragraph")
)




## === cell 7
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
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
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
    "you'll": "you you will",
    "you'll've": "you you will have",
    "you're": "you are",
    "you've": "you have",
}
c_re = re.compile("(%s)" % "|".join(cList.keys()))


def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]

    return c_re.sub(replace, text)


def removeHTML(x):
    html = re.compile(r"<.*?>")
    return html.sub(r"", x)


def dataPreprocessing(x):
    x = x.lower()
    x = removeHTML(x)
    x = re.sub("@\w+", "", x)
    x = re.sub("'\d+", "", x)
    x = re.sub("\d+", "", x)
    x = re.sub("http\w+", "", x)
    x = re.sub(r"\s+", " ", x)
    x = expandContractions(x)
    x = re.sub(r"\.+", ".", x)
    x = re.sub(r"\,+", ",", x)
    x = re.sub(r'[^\w\s.,;:"\'?!]', "", x)
    x = x.strip()
    return x




## === cell 8
class DummySpellChecker:
    def unknown(self, words):
        return set()  # no misspellings detected


spell = DummySpellChecker()


def count_misspelled_words(text):
    return 0




## === cell 9
import nltk

nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")


def count_stop_words(text):
    tokens = [token for token in text.split() if token in stop_words]
    return len(tokens)


def Cleaning(text):
    tokens = [token for token in text.split() if token not in stop_words]
    return " ".join(tokens)




## === cell 10
paragraph_features = [
    "paragraph_len",
    "paragraph_sentence_cnt",
    "paragraph_word_cnt",
    "paragraph_comma_cnt",
    "paragraph_misspelled_cnt",
    "paragraph_uni_word_cnt",
]


def Paragraph_Features(x):
    x = x.explode("paragraph")
    print("Paragraph Preprocessing")
    x = x.with_columns(pl.col("paragraph").map_elements(dataPreprocessing))
    print("Calculate the length of each paragraph")
    x = x.with_columns(
        pl.col("paragraph").map_elements(lambda s: len(s)).alias("paragraph_len")
    )
    x = x.with_columns(
        pl.col("paragraph")
        .map_elements(lambda s: count_misspelled_words(s))
        .alias("paragraph_misspelled_cnt")
    )
    x = x.with_columns(
        pl.col("paragraph")
        .map_elements(lambda s: s.count(","))
        .alias("paragraph_comma_cnt")
    )
    print("Calculate the number of sentences and words in each paragraph")
    x = x.with_columns(
        pl.col("paragraph")
        .map_elements(lambda s: len(s.split(".")))
        .alias("paragraph_sentence_cnt"),
        pl.col("paragraph")
        .map_elements(lambda s: len(s.split(" ")))
        .alias("paragraph_word_cnt"),
        pl.col("paragraph")
        .map_elements(lambda s: len(set(s.split(" "))))
        .alias("paragraph_uni_word_cnt"),
    )
    return x


def Paragraph_aggregation(x):
    try:
        print("Aggregation")
        aggs = [
            *[
                pl.col("paragraph")
                .filter(pl.col("paragraph_len") >= i)
                .count()
                .alias(f"paragraph_{i}_cnt")
                for i in [100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 800]
            ],
            *[
                pl.col("paragraph")
                .filter(pl.col("paragraph_len") <= i)
                .count()
                .alias(f"paragraph_{i}_cnt_v2")
                for i in [100, 200]
            ],
            pl.col("paragraph")
            .filter((pl.col("paragraph_len") <= 300) & (pl.col("paragraph_len") > 100))
            .count()
            .alias("short_paragraph_cnt"),
            pl.col("paragraph")
            .filter((pl.col("paragraph_len") <= 500) & (pl.col("paragraph_len") > 300))
            .count()
            .alias("mid_paragraph_cnt"),
            pl.col("paragraph")
            .filter((pl.col("paragraph_len") <= 700) & (pl.col("paragraph_len") > 500))
            .count()
            .alias("long_paragraph_cnt"),
            *[
                pl.col("paragraph")
                .filter(pl.col("paragraph_sentence_cnt") >= i)
                .count()
                .alias(f"paragraph_sentence_{i}_cnt")
                for i in [2, 4, 6, 8, 10]
            ],
            pl.col("paragraph")
            .filter(
                (pl.col("paragraph_sentence_cnt") <= 4)
                & (pl.col("paragraph_sentence_cnt") > 2)
            )
            .count()
            .alias("short_paragraph_sentence_cnt"),
            pl.col("paragraph")
            .filter(
                (pl.col("paragraph_sentence_cnt") <= 8)
                & (pl.col("paragraph_sentence_cnt") > 4)
            )
            .count()
            .alias("mid_paragraph_sentence_cnt"),
            pl.col("paragraph")
            .filter(
                (pl.col("paragraph_sentence_cnt") <= 10)
                & (pl.col("paragraph_sentence_cnt") > 8)
            )
            .count()
            .alias("long_paragraph_sentence_cnt"),
            *[
                pl.col("paragraph")
                .filter(pl.col("paragraph_word_cnt") >= i)
                .count()
                .alias(f"paragraph_word_{i}_cnt")
                for i in [20, 40, 60, 90, 120]
            ],
            pl.col("paragraph")
            .filter(pl.col("paragraph_word_cnt") <= 40)
            .filter(pl.col("paragraph_word_cnt") > 20)
            .count()
            .alias("short_paragraph_word_cnt"),
            pl.col("paragraph")
            .filter(pl.col("paragraph_word_cnt") <= 90)
            .filter(pl.col("paragraph_word_cnt") > 40)
            .count()
            .alias("mid_paragraph_word_cnt"),
            pl.col("paragraph")
            .filter(pl.col("paragraph_word_cnt") <= 120)
            .filter(pl.col("paragraph_word_cnt") > 90)
            .count()
            .alias("long_paragraph_word_cnt"),
            *[
                pl.col("paragraph")
                .filter(pl.col("paragraph_comma_cnt") >= i)
                .count()
                .alias(f"paragraph_comma_{i}_cnt")
                for i in [1, 2, 3, 4, 5]
            ],
            *[
                pl.col("paragraph")
                .filter(pl.col("paragraph_misspelled_cnt") >= i)
                .count()
                .alias(f"paragraph_misspelled_{i}_cnt")
                for i in [4, 8, 12, 16]
            ],
            *[
                pl.col("paragraph")
                .filter(pl.col("paragraph_misspelled_cnt") <= i)
                .count()
                .alias(f"paragraph_misspelled_{i}_cnt_v2")
                for i in [2, 4]
            ],
            pl.col("paragraph")
            .filter(
                (pl.col("paragraph_misspelled_cnt") <= 8)
                & (pl.col("paragraph_misspelled_cnt") > 4)
            )
            .count()
            .alias("short_paragraph_misspelled_cnt"),
            pl.col("paragraph")
            .filter(
                (pl.col("paragraph_misspelled_cnt") <= 12)
                & (pl.col("paragraph_misspelled_cnt") > 8)
            )
            .count()
            .alias("mid_paragraph_misspelled_cnt"),
            pl.col("paragraph")
            .filter(
                (pl.col("paragraph_misspelled_cnt") <= 16)
                & (pl.col("paragraph_misspelled_cnt") > 12)
            )
            .count()
            .alias("long_paragraph_misspelled_cnt"),
            pl.col("paragraph").count().alias("paragraph_cnt"),
            *[pl.col(feat).max().alias(f"{feat}_max") for feat in paragraph_features],
            *[pl.col(feat).mean().alias(f"{feat}_mean") for feat in paragraph_features],
            *[pl.col(feat).min().alias(f"{feat}_min") for feat in paragraph_features],
            *[pl.col(feat).std().alias(f"{feat}_std") for feat in paragraph_features],
            *[pl.col(feat).sum().alias(f"{feat}_sum") for feat in paragraph_features],
            *[
                pl.col(feat).quantile(0.25).alias(f"{feat}_q1")
                for feat in paragraph_features
            ],
            *[
                pl.col(feat).quantile(0.75).alias(f"{feat}_q3")
                for feat in paragraph_features
            ],
        ]
        df = x.group_by(["essay_id"], maintain_order=True).agg(aggs).sort("essay_id")
        return df.to_pandas()
    except Exception as e:
        print("Paragraph aggregation failed:", e)
        return pd.DataFrame({"essay_id": x["essay_id"].unique()})




## === cell 11
sentence_features = ["sentence_len", "sentence_word_cnt", "sentence_misspelled_cnt"]


def Sentence_Features(x):
    print("Preprocess full_text and segment sentences")
    x = x.with_columns(
        pl.col("full_text")
        .map_elements(dataPreprocessing)
        .str.split(".")
        .alias("sentence")
    )
    x = x.explode("sentence")
    print("Calculate sentence length")
    x = x.with_columns(
        pl.col("sentence").map_elements(lambda s: len(s)).alias("sentence_len")
    )
    x = x.filter(pl.col("sentence_len") > 3)
    x = x.with_columns(
        pl.col("sentence")
        .map_elements(lambda s: count_misspelled_words(s))
        .alias("sentence_misspelled_cnt")
    )
    x = x.with_columns(
        pl.col("sentence")
        .map_elements(lambda s: len(s.replace(" ", "")))
        .alias("only_sentence_len")
    )
    print("Count words in each sentence")
    x = x.with_columns(
        pl.col("sentence")
        .map_elements(lambda s: len(s.split(" ")))
        .alias("sentence_word_cnt")
    )
    return x


def Sentence_aggregation(x):
    try:
        print("Aggregation")
        aggs = [
            *[
                pl.col("sentence")
                .filter(pl.col("sentence_len") >= i)
                .count()
                .alias(f"sentence_{i}_cnt")
                for i in [40, 60, 70, 80, 100, 120, 140]
            ],
            *[
                pl.col("sentence")
                .filter(pl.col("sentence_len") <= i)
                .count()
                .alias(f"sentence_{i}_cnt_v2")
                for i in [10, 20, 30]
            ],
            pl.col("sentence")
            .filter((pl.col("sentence_len") <= 70) & (pl.col("sentence_len") > 40))
            .count()
            .alias("short_sentence_cnt"),
            pl.col("sentence")
            .filter((pl.col("sentence_len") <= 100) & (pl.col("sentence_len") > 70))
            .count()
            .alias("mid_sentence_cnt"),
            pl.col("sentence")
            .filter((pl.col("sentence_len") <= 140) & (pl.col("sentence_len") > 100))
            .count()
            .alias("long_sentence_cnt"),
            *[
                pl.col("sentence")
                .filter(pl.col("sentence_sentence_cnt") >= i)
                .count()
                .alias(f"sentence_sentence_{i}_cnt")
                for i in [2, 4, 6, 8, 10]
            ],
            pl.col("sentence")
            .filter(
                (pl.col("sentence_sentence_cnt") <= 4)
                & (pl.col("sentence_sentence_cnt") > 2)
            )
            .count()
            .alias("short_sentence_sentence_cnt"),
            pl.col("sentence")
            .filter(
                (pl.col("sentence_sentence_cnt") <= 8)
                & (pl.col("sentence_sentence_cnt") > 4)
            )
            .count()
            .alias("mid_sentence_sentence_cnt"),
            pl.col("sentence")
            .filter(
                (pl.col("sentence_sentence_cnt") <= 10)
                & (pl.col("sentence_sentence_cnt") > 8)
            )
            .count()
            .alias("long_sentence_sentence_cnt"),
            *[
                pl.col("sentence")
                .filter(pl.col("sentence_word_cnt") >= i)
                .count()
                .alias(f"sentence_word_{i}_cnt")
                for i in [10, 15, 20, 25]
            ],
            pl.col("sentence")
            .filter(
                (pl.col("sentence_word_cnt") <= 15) & (pl.col("sentence_word_cnt") > 10)
            )
            .count()
            .alias("short_sentence_word_cnt"),
            pl.col("sentence")
            .filter(
                (pl.col("sentence_word_cnt") <= 20) & (pl.col("sentence_word_cnt") > 15)
            )
            .count()
            .alias("mid_sentence_word_cnt"),
            pl.col("sentence")
            .filter(
                (pl.col("sentence_word_cnt") <= 25) & (pl.col("sentence_word_cnt") > 20)
            )
            .count()
            .alias("long_sentence_word_cnt"),
            pl.col("sentence").count().alias("sentence_cnt"),
            *[pl.col(feat).max().alias(f"{feat}_max") for feat in sentence_features],
            *[pl.col(feat).mean().alias(f"{feat}_mean") for feat in sentence_features],
            *[pl.col(feat).min().alias(f"{feat}_min") for feat in sentence_features],
            *[pl.col(feat).std().alias(f"{feat}_std") for feat in sentence_features],
            *[pl.col(feat).sum().alias(f"{feat}_sum") for feat in sentence_features],
            *[
                pl.col(feat).quantile(0.25).alias(f"{feat}_q1")
                for feat in sentence_features
            ],
            *[
                pl.col(feat).quantile(0.75).alias(f"{feat}_q3")
                for feat in sentence_features
            ],
            *[
                (pl.col(f"sentence_{i}_cnt") / pl.col("sentence_cnt")).alias(
                    f"sentence_{i}_cnt_ratio"
                )
                for i in [40, 60, 70, 80, 100, 120, 140]
            ],
            (pl.col("short_sentence_cnt") / pl.col("sentence_cnt")).alias(
                "short_sentence_cnt_ratio"
            ),
            (pl.col("mid_sentence_cnt") / pl.col("sentence_cnt")).alias(
                "mid_sentence_cnt_ratio"
            ),
            (pl.col("long_sentence_cnt") / pl.col("sentence_cnt")).alias(
                "long_sentence_cnt_ratio"
            ),
        ]
        df = x.group_by(["essay_id"], maintain_order=True).agg(aggs).sort("essay_id")
        return df.to_pandas()
    except Exception as e:
        print("Sentence aggregation failed:", e)
        return pd.DataFrame({"essay_id": x["essay_id"].unique()})




## === cell 12
word_features = ["word_len"]


def Word_Features(x):
    print("Preprocess full_text and split words")
    x = x.with_columns(
        pl.col("full_text").map_elements(dataPreprocessing).str.split(" ").alias("word")
    )
    x = x.explode("word")
    print("Calculate word length")
    x = x.with_columns(pl.col("word").map_elements(lambda w: len(w)).alias("word_len"))
    x = x.filter(pl.col("word_len") > 0)
    return x


def Word_aggregation(x):
    try:
        print("Aggregation")
        aggs = [
            *[
                pl.col("word")
                .filter(pl.col("word_len") >= i)
                .count()
                .alias(f"word_{i}_cnt")
                for i in [3, 4, 5, 6, 7, 8, 10]
            ],
            *[
                pl.col("word")
                .filter(pl.col("word_len") <= i)
                .count()
                .alias(f"word_{i}_cnt_v2")
                for i in [1, 2, 3]
            ],
            pl.col("word")
            .filter((pl.col("word_len") <= 4) & (pl.col("word_len") > 2))
            .count()
            .alias("short_word_cnt"),
            pl.col("word")
            .filter((pl.col("word_len") <= 6) & (pl.col("word_len") > 4))
            .count()
            .alias("mid_word_cnt"),
            pl.col("word")
            .filter((pl.col("word_len") <= 10) & (pl.col("word_len") > 6))
            .count()
            .alias("long_word_cnt"),
            pl.col("word").count().alias("word_cnt"),
            *[pl.col(feat).max().alias(f"{feat}_max") for feat in word_features],
            *[pl.col(feat).mean().alias(f"{feat}_mean") for feat in word_features],
            *[pl.col(feat).min().alias(f"{feat}_min") for feat in word_features],
            *[pl.col(feat).std().alias(f"{feat}_std") for feat in word_features],
            *[pl.col(feat).sum().alias(f"{feat}_sum") for feat in word_features],
            *[
                pl.col(feat).quantile(0.25).alias(f"{feat}_q1")
                for feat in word_features
            ],
            *[
                pl.col(feat).quantile(0.75).alias(f"{feat}_q3")
                for feat in word_features
            ],
            *[
                (pl.col(f"word_{i}_cnt") / pl.col("word_cnt")).alias(
                    f"word_{i}_cnt_ratio"
                )
                for i in [3, 4, 5, 6, 7, 8, 10]
            ],
            *[
                (pl.col(f"word_{i}_cnt_v2") / pl.col("word_cnt")).alias(
                    f"word_{i}_cnt_v2_ratio"
                )
                for i in [1, 2, 3]
            ],
            *[
                (pl.col(f"word_{i}_cnt") / pl.col("word_2_cnt_v2")).alias(
                    f"word_{i}_pre2_ratio"
                )
                for i in [3, 4, 5, 6, 7, 8, 10]
            ],
            *[
                (pl.col(f"word_{i}_cnt") / pl.col("word_3_cnt_v2")).alias(
                    f"word_{i}_pre3_ratio"
                )
                for i in [3, 4, 5, 6, 7, 8, 10]
            ],
            *[
                (pl.col("short_word_cnt") / pl.col(f"word_{i}_cnt_v2")).alias(
                    f"short_word_ratio_{i}"
                )
                for i in [1, 2, 3]
            ],
            *[
                (pl.col("mid_word_cnt") / pl.col(f"word_{i}_cnt_v2")).alias(
                    f"mid_word_ratio_{i}"
                )
                for i in [1, 2, 3]
            ],
            *[
                (pl.col("long_word_cnt") / pl.col(f"word_{i}_cnt_v2")).alias(
                    f"long_word_ratio_{i}"
                )
                for i in [1, 2, 3]
            ],
        ]
        df = x.group_by(["essay_id"], maintain_order=True).agg(aggs).sort("essay_id")
        return df.to_pandas()
    except Exception as e:
        print("Word aggregation failed:", e)
        return pd.DataFrame({"essay_id": x["essay_id"].unique()})




## === cell 13
vectorizer = TfidfVectorizer(
    tokenizer=lambda x: x,
    preprocessor=lambda x: x,
    token_pattern=None,
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(1, 4),
    min_df=0.01,  # changed from 0.05
    max_df=0.90,  # changed from 0.95
    sublinear_tf=True,
)

train_a = train.with_columns(pl.col("full_text").map_elements(dataPreprocessing))
train_tfid = vectorizer.fit_transform([i for i in train_a["full_text"]])

dense_matrix = train_tfid.toarray()
df = pd.DataFrame(dense_matrix)
df.columns = [f"tfidf_{i}" for i in range(len(df.columns))]
df["essay_id"] = df_train["essay_id"]




## === cell 14
vectorizer_cnt = CountVectorizer(
    tokenizer=lambda x: x,
    preprocessor=lambda x: x,
    token_pattern=None,
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(2, 3),
    min_df=0.01,  # changed from 0.10
    max_df=0.90,  # changed from 0.80
)

train_b = train.with_columns(pl.col("full_text").map_elements(dataPreprocessing))
train_b = train_b.with_columns(pl.col("full_text").map_elements(lambda s: Cleaning(s)))
train_cnt = vectorizer_cnt.fit_transform([i for i in train_b["full_text"]])

dense_matrix2 = train_cnt.toarray()
df2 = pd.DataFrame(dense_matrix2)
df2.columns = [f"cnt_{i}" for i in range(len(df2.columns))]
df2["essay_id"] = df_train["essay_id"]




## === cell 15
try:
    train_feats1 = Paragraph_Features(train)
    train_feats1 = Paragraph_aggregation(train_feats1)
except Exception as e:
    print("Paragraph features failed:", e)
    train_feats1 = pd.DataFrame({"essay_id": df_train["essay_id"].unique()})

try:
    train_feats2 = Sentence_Features(train)
    train_feats2 = Sentence_aggregation(train_feats2)
except Exception as e:
    print("Sentence features failed:", e)
    train_feats2 = pd.DataFrame({"essay_id": df_train["essay_id"].unique()})

try:
    train_feats3 = Word_Features(train)
    train_feats3 = Word_aggregation(train_feats3)
except Exception as e:
    print("Word features failed:", e)
    train_feats3 = pd.DataFrame({"essay_id": df_train["essay_id"].unique()})

train_feats = train_feats1.merge(train_feats2, on="essay_id", how="left")
train_feats = train_feats.merge(train_feats3, on="essay_id", how="left")
train_feats = train_feats.merge(df, on="essay_id", how="left")
train_feats = train_feats.merge(df2, on="essay_id", how="left")
train_feats["score"] = df_train["score"].values

print("Engineered train features shape:", train_feats.shape)




## === cell 16
print("Saving engineered train features")
train_feats.to_csv(f"train_feats_{CFG.VER}.csv", index=False)




## === cell 17
display(train_feats.head())




## === cell 18
print("LightGBM Version:", lgb.__version__)




## === cell 19
def quadratic_weighted_kappa(y_true, y_pred):
    y_true = y_true + a
    y_pred = (y_pred + a).clip(1, 6).round()
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return "QWK", qwk, True


def qwk_obj(y_true, y_pred):
    labels = y_true + a
    preds = y_pred + a
    preds = preds.clip(1, 6)
    f = 0.5 * np.sum((preds - labels) ** 2)
    g = 0.5 * np.sum((preds - a) ** 2 + b)
    df = preds - labels
    dg = preds - a
    grad = (df / g - f * dg / g**2) * len(labels)
    hess = np.ones(len(labels))
    return grad, hess


a = 2.948
b = 1.092




## === cell 20
categorical_columns = train_feats.select_dtypes(
    include=["object", "category"]
).columns.tolist()
FEATURES = [
    col for col in train_feats.columns if col not in categorical_columns + ["score"]
]
TARGET = "score"




## === cell 21
all_oof = []
all_true = []

skf = StratifiedKFold(n_splits=15, random_state=CFG.SEED, shuffle=True)
for i, (train_index, valid_index) in enumerate(
    skf.split(train_feats, train_feats[TARGET])
):
    print("#" * 25)
    print(f"### Fold {i+1}")
    print(f"### train size {len(train_index)}, valid size {len(valid_index)}")
    print("#" * 25)

    model = lgb.LGBMRegressor(
        objective=qwk_obj,
        learning_rate=0.05,
        colsample_bytree=0.8,
        max_depth=5,
        num_leaves=10,
        reg_alpha=0.2,
        reg_lambda=0.8,
        n_estimators=1024,
        class_weight="balanced",
        random_state=CFG.SEED,
        verbosity=-1,
    )

    train_x = np.clip(train_feats.loc[train_index, FEATURES].fillna(0), 0, 10000)
    train_y = train_feats.loc[train_index, TARGET] - a

    valid_x = np.clip(train_feats.loc[valid_index, FEATURES].fillna(0), 0, 10000)
    valid_y = train_feats.loc[valid_index, TARGET] - a

    model.fit(
        train_x,
        train_y,
        eval_set=[(valid_x, valid_y)],
        eval_metric=quadratic_weighted_kappa,
        callbacks=[early_stopping(stopping_rounds=100)],
    )

    pickle.dump(model, open(f"LGB_v{CFG.VER}_f{i}.pkl", "wb"))

    oof = model.predict(valid_x, num_iteration=model.best_iteration_) + a
    all_oof.append(oof)
    all_true.append(valid_y.values + a)

    del train_x, train_y, valid_x, valid_y, oof, model
    clean_memory()

all_oof = np.concatenate(all_oof)
all_true = np.concatenate(all_true)

cv_score = cohen_kappa_score(
    all_true, np.clip(all_oof, 1, 6).round(), weights="quadratic"
)
print("CV Score for LightGBM =", cv_score)




## === cell 22
def optimize_threshold(true, pred, steps=100):
    threshold = [1.5, 2.5, 3.5, 4.5, 5.5]
    try:
        trial_x = [[] for _ in range(5)]
        trial_y = [[] for _ in range(5)]
        best = cohen_kappa_score(
            true,
            pd.cut(
                pred, [-np.inf] + threshold + [np.inf], labels=[1, 2, 3, 4, 5, 6]
            ).astype("int32"),
            weights="quadratic",
        )
        for k in range(5):
            for sign in [1, -1]:
                v = threshold[k]
                thresh_copy = threshold.copy()
                stop = 0
                while stop < steps:
                    v += sign * 0.001
                    thresh_copy[k] = v
                    if not all(
                        x < y
                        for x, y in zip([-np.inf] + thresh_copy, thresh_copy + [np.inf])
                    ):
                        break
                    pred_bins = pd.cut(
                        pred,
                        [-np.inf] + thresh_copy + [np.inf],
                        labels=[1, 2, 3, 4, 5, 6],
                    ).astype("int32")
                    metric = cohen_kappa_score(true, pred_bins, weights="quadratic")
                    trial_x[k].append(v)
                    trial_y[k].append(metric)
                    if metric <= best:
                        stop += 1
                    else:
                        stop = 0
                        best = metric
                        threshold = thresh_copy.copy()
    except Exception as e:
        print("Threshold optimisation failed or produced invalid bins:", e)
    return best, threshold, trial_x, trial_y


best, threshold, _, _ = optimize_threshold(all_true, all_oof)
print("Optimized thresholds (or defaults if optimisation failed):", threshold)
print("Best CV QWK after threshold optimization =", best)




## === cell 23
test_a = test.with_columns(pl.col("full_text").map_elements(dataPreprocessing))
test_tfid = vectorizer.transform([i for i in test_a["full_text"]])
df3 = pd.DataFrame(test_tfid.toarray())
df3.columns = [f"tfidf_{i}" for i in range(len(df3.columns))]
df3["essay_id"] = df_test["essay_id"]




## === cell 24
test_b = test.with_columns(pl.col("full_text").map_elements(dataPreprocessing))
test_b = test_b.with_columns(pl.col("full_text").map_elements(lambda s: Cleaning(s)))
test_cnt = vectorizer_cnt.transform([i for i in test_b["full_text"]])
df4 = pd.DataFrame(test_cnt.toarray())
df4.columns = [f"cnt_{i}" for i in range(len(df4.columns))]
df4["essay_id"] = df_test["essay_id"]




## === cell 25
try:
    test_feats1 = Paragraph_Features(test)
    test_feats1 = Paragraph_aggregation(test_feats1)
except Exception as e:
    print("Test paragraph features failed:", e)
    test_feats1 = pd.DataFrame({"essay_id": df_test["essay_id"].unique()})

try:
    test_feats2 = Sentence_Features(test)
    test_feats2 = Sentence_aggregation(test_feats2)
except Exception as e:
    print("Test sentence features failed:", e)
    test_feats2 = pd.DataFrame({"essay_id": df_test["essay_id"].unique()})

try:
    test_feats3 = Word_Features(test)
    test_feats3 = Word_aggregation(test_feats3)
except Exception as e:
    print("Test word features failed:", e)
    test_feats3 = pd.DataFrame({"essay_id": df_test["essay_id"].unique()})




## === cell 26
test_feats = test_feats1.merge(test_feats2, on="essay_id", how="left")
test_feats = test_feats.merge(test_feats3, on="essay_id", how="left")
test_feats = test_feats.merge(df3, on="essay_id", how="left")
test_feats = test_feats.merge(df4, on="essay_id", how="left")

print("Test features shape:", test_feats.shape)




## === cell 27
preds = []
categorical_columns_test = test_feats.select_dtypes(
    include=["object", "category"]
).columns.tolist()
FEATURES_TEST = [
    col for col in test_feats.columns if col not in categorical_columns_test
]

for i in range(15):
    print(f"Fold {i+1}")
    model = pickle.load(open(f"LGB_v{CFG.VER}_f{i}.pkl", "rb"))
    pred = model.predict(test_feats[FEATURES_TEST]) + a
    preds.append(pred)

pred1 = np.mean(preds, axis=0)




## === cell 28
submission = pd.DataFrame({"essay_id": df_test["essay_id"]})
submission["score"] = pd.cut(
    pred1, [-np.inf] + threshold + [np.inf], labels=[1, 2, 3, 4, 5, 6]
).astype("int32")
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv – shape:", submission.shape)
submission.head()
