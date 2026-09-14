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
import random
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import scipy.sparse as sp
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
import lightgbm as lgb
from lightgbm import early_stopping
from pathlib import Path


class CFG:
    SEED = 42
    VER = "v1"
    _possible_paths = [
        Path("./data/learning-agency-lab-automated-essay-scoring-2/"),
        Path("./data/input/learning-agency-lab-automated-essay-scoring-2/"),
        Path("./input/learning-agency-lab-automated-essay-scoring-2/"),
        Path("./working/learning-agency-lab-automated-essay-scoring-2/"),
        Path("/kaggle/input/learning-agency-lab-automated-essay-scoring-2/"),
    ]

    for p in _possible_paths:
        if p.is_dir():
            BASE_PATH = str(p) + "/"
            break
    else:
        raise FileNotFoundError(
            "Dataset directory not found. Checked: "
            + ", ".join(str(p) for p in _possible_paths)
        )


def clean_memory():
    """Placeholder for any memory cleanup; kept lightweight."""
    pass




## === cell 1
def seed_everything():
    random.seed(CFG.SEED)
    np.random.seed(CFG.SEED)
    os.environ["PYTHONHASHSEED"] = str(CFG.SEED)


seed_everything()



## === cell 2
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
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "o'clock": "of the clock",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "so've": "so have",
    "that's": "that is",
    "there'd": "there had",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what's": "what is",
    "when's": "when is",
    "where'd": "where did",
    "where's": "where is",
    "who's": "who is",
    "won't": "will not",
    "would've": "would have",
    "wouldn't": "would not",
    "y'all": "you all",
    "you'd": "you had",
    "you'll": "you will",
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
    x = re.sub("@\\w+", "", x)
    x = re.sub("'\\d+", "", x)
    x = re.sub("\\d+", "", x)
    x = re.sub("http\\w+", "", x)
    x = re.sub(r"\s+", " ", x)
    x = expandContractions(x)
    x = re.sub(r"\.+", ".", x)
    x = re.sub(r",+", ",", x)
    x = re.sub(r'[^\w\s.,;:"\'?!]', "", x)
    x = x.strip()
    return x




## === cell 3
df_train = pd.read_csv(CFG.BASE_PATH + "train.csv")
df_train = df_train.sort_values(by="essay_id")
df_train["clean_text"] = df_train["full_text"].apply(dataPreprocessing)

df_test = pd.read_csv(CFG.BASE_PATH + "test.csv")
df_test = df_test.sort_values(by="essay_id")
df_test["clean_text"] = df_test["full_text"].apply(dataPreprocessing)

print("Shape of Train:", df_train.shape)
print("Shape of Test :", df_test.shape)



## === cell 4
plt.figure(figsize=(12, 6))
sns.countplot(x=df_train["score"])
plt.title("Distribution of Score")
plt.xlabel("Score of Essay")
plt.ylabel("Frequency")
plt.show()



## === cell 5
train = pl.from_pandas(df_train).with_columns(
    pl.col("full_text").str.split(by="\n\n").alias("paragraph")
)
test = pl.from_pandas(df_test).with_columns(
    pl.col("full_text").str.split(by="\n\n").alias("paragraph")
)




## === cell 6
class DummySpellChecker:
    def unknown(self, words):
        return set()


spell = DummySpellChecker()


def count_misspelled_words(text):
    return 0




## === cell 7
nltk.download("stopwords", quiet=True)
stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")


def count_stop_words(text):
    tokens = [token for token in text.split() if token in stop_words]
    return len(tokens)


def Cleaning(text):
    tokens = [token for token in text.split() if token not in stop_words]
    return " ".join(tokens)




## === cell 8
paragraph_features = [
    "paragraph_len",
    "paragraph_sentence_cnt",
    "paragraph_word_cnt",
    "paragraph_comma_cnt",
    "paragraph_misspelled_cnt",
    "paragraph_uni_word_cnt",
]


def Paragraph_Features(x):
    x = x.with_columns(pl.col("clean_text").str.split("\n\n").alias("paragraph"))
    x = x.explode("paragraph")
    x = x.with_columns(pl.col("paragraph").str.lengths().alias("paragraph_len"))
    x = x.with_columns(pl.lit(0).alias("paragraph_misspelled_cnt"))
    x = x.with_columns(
        pl.col("paragraph").str.count_match(",").alias("paragraph_comma_cnt")
    )
    x = x.with_columns(
        pl.col("paragraph").str.split(".").list.len().alias("paragraph_sentence_cnt"),
        pl.col("paragraph").str.split(" ").list.len().alias("paragraph_word_cnt"),
        pl.col("paragraph")
        .str.split(" ")
        .list.unique()
        .list.len()
        .alias("paragraph_uni_word_cnt"),
    )
    return x


def Paragraph_aggregation(x):
    try:
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




## === cell 9
sentence_features = ["sentence_len", "sentence_word_cnt", "sentence_misspelled_cnt"]


def Sentence_Features(x):
    x = x.with_columns(pl.col("clean_text").str.split(".").alias("sentence"))
    x = x.explode("sentence")
    x = x.with_columns(pl.col("sentence").str.lengths().alias("sentence_len"))
    x = x.filter(pl.col("sentence_len") > 3)
    x = x.with_columns(pl.lit(0).alias("sentence_misspelled_cnt"))
    x = x.with_columns(
        pl.col("sentence")
        .str.replace_all(" ", "")
        .str.lengths()
        .alias("only_sentence_len")
    )
    x = x.with_columns(
        pl.col("sentence").str.split(" ").list.len().alias("sentence_word_cnt")
    )
    return x


def Sentence_aggregation(x):
    try:
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
            .filter(pl.col("sentence_word_cnt") <= 15)
            .filter(pl.col("sentence_word_cnt") > 10)
            .count()
            .alias("short_sentence_word_cnt"),
            pl.col("sentence")
            .filter(pl.col("sentence_word_cnt") <= 20)
            .filter(pl.col("sentence_word_cnt") > 15)
            .count()
            .alias("mid_sentence_word_cnt"),
            pl.col("sentence")
            .filter(pl.col("sentence_word_cnt") <= 25)
            .filter(pl.col("sentence_word_cnt") > 20)
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




## === cell 10
def Word_Features(x):
    x = x.with_columns(pl.col("clean_text").str.split(" ").alias("word"))
    x = x.explode("word")
    x = x.with_columns(pl.col("word").str.lengths().alias("word_len"))
    x = x.filter(pl.col("word_len") > 0)
    return x


def Word_aggregation(x):
    try:
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
            pl.col("word_len").max().alias("word_len_max"),
            pl.col("word_len").mean().alias("word_len_mean"),
            pl.col("word_len").min().alias("word_len_min"),
            pl.col("word_len").std().alias("word_len_std"),
            pl.col("word_len").sum().alias("word_len_sum"),
            pl.col("word_len").quantile(0.25).alias("word_len_q1"),
            pl.col("word_len").quantile(0.75).alias("word_len_q3"),
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
        ]
        df = x.group_by(["essay_id"], maintain_order=True).agg(aggs).sort("essay_id")
        return df.to_pandas()
    except Exception as e:
        print("Word aggregation failed:", e)
        return pd.DataFrame({"essay_id": x["essay_id"].unique()})




## === cell 11
vectorizer = TfidfVectorizer(
    tokenizer=lambda x: x,
    preprocessor=lambda x: x,
    token_pattern=None,
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(1, 4),
    min_df=0.01,
    max_df=0.90,
    sublinear_tf=True,
)

tfidf_train = vectorizer.fit_transform(df_train["clean_text"].tolist())

vectorizer_cnt = CountVectorizer(
    tokenizer=lambda x: x,
    preprocessor=lambda x: x,
    token_pattern=None,
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(2, 3),
    min_df=0.01,
    max_df=0.90,
)

train_b_clean = df_train["clean_text"].apply(Cleaning)
cnt_train = vectorizer_cnt.fit_transform(train_b_clean.tolist())



## === cell 12
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
train_feats["score"] = df_train["score"].values

print("Engineered train features shape (without TF-IDF/Count):", train_feats.shape)



## === cell 13
train_feats.to_csv(f"train_feats_{CFG.VER}.csv", index=False)



## === cell 14
print("LightGBM Version:", lgb.__version__)




## === cell 15
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



## === cell 16
categorical_columns = train_feats.select_dtypes(
    include=["object", "category"]
).columns.tolist()
FEATURES = [
    col for col in train_feats.columns if col not in categorical_columns + ["score"]
]
TARGET = "score"



## === cell 17
dense_all = np.clip(train_feats[FEATURES].fillna(0).values, 0, 10000)
dense_csr_all = sp.csr_matrix(dense_all)

full_train_csr = sp.hstack([dense_csr_all, tfidf_train, cnt_train], format="csr")

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

    train_x = full_train_csr[train_index]
    valid_x = full_train_csr[valid_index]

    train_y = train_feats.loc[train_index, TARGET] - a
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

    del train_x, valid_x, train_y, valid_y, oof, model
    clean_memory()

all_oof = np.concatenate(all_oof)
all_true = np.concatenate(all_true)

cv_score = cohen_kappa_score(
    all_true, np.clip(all_oof, 1, 6).round(), weights="quadratic"
)
print("CV Score for LightGBM =", cv_score)




## === cell 18
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



## === cell 19
tfidf_test = vectorizer.transform(df_test["clean_text"].tolist())

test_b_clean = df_test["clean_text"].apply(Cleaning)
cnt_test = vectorizer_cnt.transform(test_b_clean.tolist())



## === cell 20
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



## === cell 21
test_feats = test_feats1.merge(test_feats2, on="essay_id", how="left")
test_feats = test_feats.merge(test_feats3, on="essay_id", how="left")
print("Test features shape (numeric only):", test_feats.shape)



## === cell 22
dense_test_all = np.clip(test_feats[FEATURES].fillna(0).values, 0, 10000)
dense_test = sp.csr_matrix(dense_test_all)

full_test_csr = sp.hstack([dense_test, tfidf_test, cnt_test], format="csr")

preds = []
for i in range(15):
    print(f"Fold {i+1}")
    model = pickle.load(open(f"LGB_v{CFG.VER}_f{i}.pkl", "rb"))
    pred = model.predict(full_test_csr) + a
    preds.append(pred)

pred1 = np.mean(preds, axis=0)



## === cell 23
submission = pd.DataFrame({"essay_id": df_test["essay_id"]})
submission["score"] = pd.cut(
    pred1, [-np.inf] + threshold + [np.inf], labels=[1, 2, 3, 4, 5, 6]
).astype("int32")
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv – shape:", submission.shape)
submission.head()
