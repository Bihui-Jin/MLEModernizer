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

# 5. Target score

0.8124118061679555

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
Clean = True


def clean_memory():
    if Clean:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        gc.collect()


clean_memory()




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_57/3543351314.py in <cell line: 0>()
      8 
      9 
---> 10 clean_memory()
     11 
     12 

/tmp/ipykernel_57/3543351314.py in clean_memory()
      4 def clean_memory():
      5     if Clean:
----> 6         ctypes.CDLL("libc.so.6").malloc_trim(0)
      7         gc.collect()
      8 

NameError: name 'ctypes' is not defined

## === cell 1
def seed_everything():
    random.seed(CFG.SEED)
    np.random.seed(CFG.SEED)
    os.environ["PYTHONHASHSEED"] = str(CFG.SEED)


seed_everything()




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_57/3006590381.py in <cell line: 0>()
      5 
      6 
----> 7 seed_everything()
      8 
      9 

/tmp/ipykernel_57/3006590381.py in seed_everything()
      1 def seed_everything():
----> 2     random.seed(CFG.SEED)
      3     np.random.seed(CFG.SEED)
      4     os.environ["PYTHONHASHSEED"] = str(CFG.SEED)
      5 

NameError: name 'random' is not defined

## === cell 2
df_train = pd.read_csv(CFG.BASE_PATH + "train.csv")
df_train = df_train.sort_values(by="essay_id")
print("Shape of Train:", df_train.shape)
print(df_train.head())




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_57/2411673526.py in <cell line: 0>()
----> 1 df_train = pd.read_csv(CFG.BASE_PATH + "train.csv")
      2 df_train = df_train.sort_values(by="essay_id")
      3 print("Shape of Train:", df_train.shape)
      4 print(df_train.head())
      5 

NameError: name 'pd' is not defined

## === cell 3
plt.figure(figsize=(12, 6))
sns.countplot(x=df_train["score"])
plt.title("Distribution of Score")
plt.xlabel("Score of Essay")
plt.ylabel("Frequency")
plt.show()




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_57/2868796577.py in <cell line: 0>()
----> 1 plt.figure(figsize=(12, 6))
      2 sns.countplot(x=df_train["score"])
      3 plt.title("Distribution of Score")
      4 plt.xlabel("Score of Essay")
      5 plt.ylabel("Frequency")

NameError: name 'plt' is not defined

## === cell 4
df_test = pd.read_csv(CFG.BASE_PATH + "test.csv")
df_test = df_test.sort_values(by="essay_id")
print("Shape of Test:", df_test.shape)
print(df_test.head())




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_57/1062947332.py in <cell line: 0>()
----> 1 df_test = pd.read_csv(CFG.BASE_PATH + "test.csv")
      2 df_test = df_test.sort_values(by="essay_id")
      3 print("Shape of Test:", df_test.shape)
      4 print(df_test.head())
      5 

NameError: name 'pd' is not defined

## === cell 5
train = pl.from_pandas(df_train).with_columns(
    pl.col("full_text").str.split(by="\n\n").alias("paragraph")
)
test = pl.from_pandas(df_test).with_columns(
    pl.col("full_text").str.split(by="\n\n").alias("paragraph")
)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_57/1009259483.py in <cell line: 0>()
----> 1 train = pl.from_pandas(df_train).with_columns(
      2     pl.col("full_text").str.split(by="\n\n").alias("paragraph")
      3 )
      4 test = pl.from_pandas(df_test).with_columns(
      5     pl.col("full_text").str.split(by="\n\n").alias("paragraph")

NameError: name 'pl' is not defined

## === cell 6
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
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'll": "I will",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it had",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "o'clock": "of the clock",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'll": "we will",
    "we're": "we are",
    "weren't": "were not",
    "what's": "what is",
    "when's": "when is",
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




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_57/455821654.py in <cell line: 0>()
     70     "you've": "you have",
     71 }
---> 72 c_re = re.compile("(%s)" % "|".join(cList.keys()))
     73 
     74 

NameError: name 're' is not defined

## === cell 7
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
    x = re.sub(r'[^\w\s.,;:"\'!?]', "", x)
    return x.strip()




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_57/519616903.py in <cell line: 0>()
----> 1 def expandContractions(text, c_re=c_re):
      2     def replace(match):
      3         return cList[match.group(0)]
      4 
      5     return c_re.sub(replace, text)

NameError: name 'c_re' is not defined

## === cell 8
try:
    from spellchecker import SpellChecker

    spell = SpellChecker()

    def count_misspelled_words(text: str) -> int:
        return len(spell.unknown(text.split()))

except Exception:

    def count_misspelled_words(text: str) -> int:
        return 0




## === cell 9
paragraph_features = [
    "paragraph_len",
    "paragraph_sentence_cnt",
    "paragraph_word_cnt",
    "paragraph_comma_cnt",
    "paragraph_misspelled_cnt",
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
    )
    return x


def Paragraph_aggregation(x):
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
        .filter(
            (pl.col("paragraph_word_cnt") <= 40) & (pl.col("paragraph_word_cnt") > 20)
        )
        .count()
        .alias("short_paragraph_word_cnt"),
        pl.col("paragraph")
        .filter(
            (pl.col("paragraph_word_cnt") <= 90) & (pl.col("paragraph_word_cnt") > 40)
        )
        .count()
        .alias("mid_paragraph_word_cnt"),
        pl.col("paragraph")
        .filter(
            (pl.col("paragraph_word_cnt") <= 120) & (pl.col("paragraph_word_cnt") > 90)
        )
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




## === cell 10
sentence_features = ["sentence_len", "sentence_word_cnt", "sentence_misspelled_cnt"]


def Sentence_Features(x):
    print("Preprocess full_text and segment sentences")
    x = x.with_columns(
        pl.col("full_text")
        .map_elements(lambda s: dataPreprocessing(s))
        .str.split(".")
        .alias("sentence")
    )
    x = x.explode("sentence")
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
    x = x.with_columns(
        pl.col("sentence")
        .map_elements(lambda s: len(s.split(" ")))
        .alias("sentence_word_cnt")
    )
    return x


def Sentence_aggregation(x):
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
            .filter(pl.col("only_sentence_len") >= i)
            .count()
            .alias(f"only_sentence_{i}_cnt")
            for i in [40, 60, 80, 100, 120]
        ],
        pl.col("sentence")
        .filter(
            (pl.col("only_sentence_len") <= 60) & (pl.col("only_sentence_len") > 40)
        )
        .count()
        .alias("short_only_sentence_cnt"),
        pl.col("sentence")
        .filter(
            (pl.col("only_sentence_len") <= 100) & (pl.col("only_sentence_len") > 60)
        )
        .count()
        .alias("mid_only_sentence_cnt"),
        pl.col("sentence")
        .filter(
            (pl.col("only_sentence_len") <= 120) & (pl.col("only_sentence_len") > 100)
        )
        .count()
        .alias("long_only_sentence_cnt"),
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
    ]
    df = x.group_by(["essay_id"], maintain_order=True).agg(aggs).sort("essay_id")
    df = df.with_columns(
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
    )
    return df.to_pandas()




## === cell 11
word_features = ["word_len"]


def Word_Features(x):
    print("Preprocess full_text and split into words")
    x = x.with_columns(
        pl.col("full_text")
        .map_elements(lambda s: dataPreprocessing(s))
        .str.split(" ")
        .alias("word")
    )
    x = x.explode("word")
    x = x.with_columns(pl.col("word").map_elements(lambda w: len(w)).alias("word_len"))
    x = x.filter(pl.col("word_len") > 0)
    return x


def Word_aggregation(x):
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
        *[pl.col(feat).quantile(0.25).alias(f"{feat}_q1") for feat in word_features],
        *[pl.col(feat).quantile(0.75).alias(f"{feat}_q3") for feat in word_features],
    ]
    df = x.group_by(["essay_id"], maintain_order=True).agg(aggs).sort("essay_id")
    df = df.with_columns(
        *[
            (pl.col(f"word_{i}_cnt") / pl.col("word_cnt")).alias(f"word_{i}_cnt_ratio")
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
    )
    return df.to_pandas()




## === cell 12
vectorizer = TfidfVectorizer(
    tokenizer=lambda x: x,
    preprocessor=lambda x: x,
    token_pattern=None,
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(1, 4),
    min_df=0.05,
    max_df=0.95,
    sublinear_tf=True,
)

train_clean = train.with_columns(
    pl.col("full_text").map_elements(lambda s: dataPreprocessing(s))
)
train_tfid = vectorizer.fit_transform([i for i in train_clean["full_text"]])
tfidf_feature_names = [f"tfidf_{i}" for i in range(train_tfid.shape[1])]

vectorizer_cnt = CountVectorizer(
    tokenizer=lambda x: x,
    preprocessor=lambda x: x,
    token_pattern=None,
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(2, 4),
    min_df=0.10,
    max_df=0.85,
)

train_cnt = vectorizer_cnt.fit_transform([i for i in train_clean["full_text"]])
cnt_feature_names = [f"cnt_{i}" for i in range(train_cnt.shape[1])]




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_57/2223366246.py in <cell line: 0>()
----> 1 vectorizer = TfidfVectorizer(
      2     tokenizer=lambda x: x,
      3     preprocessor=lambda x: x,
      4     token_pattern=None,
      5     strip_accents="unicode",

NameError: name 'TfidfVectorizer' is not defined

## === cell 13
if CFG.LOAD_FEATURES_FROM is None:
    train_feats1 = Paragraph_Features(train)
    train_feats1 = Paragraph_aggregation(train_feats1)
    train_feats2 = Sentence_Features(train)
    train_feats2 = Sentence_aggregation(train_feats2)
    train_feats3 = Word_Features(train)
    train_feats3 = Word_aggregation(train_feats3)

    train_dense = train_feats1.merge(train_feats2, on="essay_id", how="left")
    train_dense = train_dense.merge(train_feats3, on="essay_id", how="left")
    train_dense = train_dense.merge(
        df_train[["essay_id", "score"]], on="essay_id", how="left"
    )
    train_dense = train_dense.sort_values("essay_id").reset_index(drop=True)

    TARGET = "score"

    dense_matrix = sp.csr_matrix(
        train_dense.drop(columns=["essay_id", "score"]).fillna(0).values
    )
    X_train_sparse = sp.hstack([dense_matrix, train_tfid, train_cnt], format="csr")
    y_train = df_train["score"].values
else:
    train_dense = pd.read_csv(CFG.LOAD_FEATURES_FROM)
    X_train_sparse = None
    y_train = None




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_57/1106155886.py in <cell line: 0>()
      1 # Build dense, sparse, and label matrices
----> 2 if CFG.LOAD_FEATURES_FROM is None:
      3     train_feats1 = Paragraph_Features(train)
      4     train_feats1 = Paragraph_aggregation(train_feats1)
      5     train_feats2 = Sentence_Features(train)

NameError: name 'CFG' is not defined

## === cell 14
print(train_dense.head())




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_57/2764309466.py in <cell line: 0>()
----> 1 print(train_dense.head())
      2 
      3 

NameError: name 'train_dense' is not defined

## === cell 15
print("LightGBM Version:", lgb.__version__)




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_57/3237529097.py in <cell line: 0>()
----> 1 print("LightGBM Version:", lgb.__version__)
      2 
      3 

NameError: name 'lgb' is not defined

## === cell 16
def quadratic_weighted_kappa(y_true, y_pred):
    y_true = y_true + a
    y_pred = (y_pred + a).clip(1, 6).round()
    return "QWK", cohen_kappa_score(y_true, y_pred, weights="quadratic"), True


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




## === cell 17
FEATURES = (
    list(train_dense.drop(columns=["essay_id", "score"]).columns)
    + tfidf_feature_names
    + cnt_feature_names
)




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_57/3424100420.py in <cell line: 0>()
      1 FEATURES = (
----> 2     list(train_dense.drop(columns=["essay_id", "score"]).columns)
      3     + tfidf_feature_names
      4     + cnt_feature_names
      5 )

NameError: name 'train_dense' is not defined

## === cell 18
def lightgbm_training():
    all_oof = []
    all_true = []

    skf = StratifiedKFold(n_splits=15, random_state=CFG.SEED, shuffle=True)
    for i, (train_idx, valid_idx) in enumerate(
        skf.split(train_dense, train_dense[TARGET])
    ):
        print("#" * 25)
        print(f"Fold {i+1} | train: {len(train_idx)} | valid: {len(valid_idx)}")
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
            random_state=CFG.SEED,
            verbosity=-1,
        )

        train_x = X_train_sparse[train_idx]
        train_y = y_train[train_idx] - a

        valid_x = X_train_sparse[valid_idx]
        valid_y = y_train[valid_idx] - a

        model.fit(
            train_x,
            train_y,
            eval_set=[(valid_x, valid_y)],
            eval_metric=quadratic_weighted_kappa,
            callbacks=[early_stopping(stopping_rounds=100)],
        )

        pickle.dump(model, open(f"LGB_v{CFG.VER}_f{i}.pkl", "wb"))

        oof = model.predict(valid_x, num_iteration=model.best_iteration_)
        all_oof.append(oof + a)
        all_true.append(valid_y + a)

        del train_x, train_y, valid_x, valid_y, oof, model
        clean_memory()

    all_oof = np.concatenate(all_oof)
    all_true = np.concatenate(all_true)

    cv_score = cohen_kappa_score(
        all_true, np.clip(all_oof, 1, 6).round(), weights="quadratic"
    )
    print("CV QWK:", cv_score)

    cm = confusion_matrix(all_true, np.clip(all_oof, 1, 6).round(), labels=range(1, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(1, 7))
    disp.plot()
    plt.show()




## === cell 19
if CFG.LOAD_MODELS_FROM is None:
    print("Training LightGBM models")
    lightgbm_training()
else:
    print("Skipping training (models pre‑loaded)")




## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_57/3345442132.py in <cell line: 0>()
----> 1 if CFG.LOAD_MODELS_FROM is None:
      2     print("Training LightGBM models")
      3     lightgbm_training()
      4 else:
      5     print("Skipping training (models pre‑loaded)")

NameError: name 'CFG' is not defined

## === cell 20
model = pickle.load(open(f"LGB_v{CFG.VER}_f0.pkl", "rb"))
df_importance = pd.DataFrame(
    {
        "features_name": FEATURES,
        "importance": model.feature_importances_,
    }
)
df_importance = df_importance.sort_values(by="importance", ascending=False)

plt.figure(figsize=(12, 6))
plt.bar(
    data=df_importance.head(30),
    x="features_name",
    height="importance",
    color="pink",
    edgecolor="black",
)
plt.title("Top 30 Feature Importances")
plt.xticks(rotation=90)
plt.show()




## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_57/1926467780.py in <cell line: 0>()
----> 1 model = pickle.load(open(f"LGB_v{CFG.VER}_f0.pkl", "rb"))
      2 df_importance = pd.DataFrame(
      3     {
      4         "features_name": FEATURES,
      5         "importance": model.feature_importances_,

NameError: name 'pickle' is not defined

## === cell 21
test_clean = test.with_columns(
    pl.col("full_text").map_elements(lambda s: dataPreprocessing(s))
)
test_tfid = vectorizer.transform([i for i in test_clean["full_text"]])
test_cnt = vectorizer_cnt.transform([i for i in test_clean["full_text"]])




## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_57/1085627062.py in <cell line: 0>()
----> 1 test_clean = test.with_columns(
      2     pl.col("full_text").map_elements(lambda s: dataPreprocessing(s))
      3 )
      4 test_tfid = vectorizer.transform([i for i in test_clean["full_text"]])
      5 test_cnt = vectorizer_cnt.transform([i for i in test_clean["full_text"]])

NameError: name 'test' is not defined

## === cell 22
if CFG.LOAD_FEATURES_FROM is None:
    test_feats1 = Paragraph_Features(test)
    test_feats1 = Paragraph_aggregation(test_feats1)
    test_feats2 = Sentence_Features(test)
    test_feats2 = Sentence_aggregation(test_feats2)
    test_feats3 = Word_Features(test)
    test_feats3 = Word_aggregation(test_feats3)

    test_dense = test_feats1.merge(test_feats2, on="essay_id", how="left")
    test_dense = test_dense.merge(test_feats3, on="essay_id", how="left")
    test_dense = test_dense.sort_values("essay_id").reset_index(drop=True)

    dense_matrix_test = sp.csr_matrix(
        test_dense.drop(columns=["essay_id"]).fillna(0).values
    )
    X_test_sparse = sp.hstack([dense_matrix_test, test_tfid, test_cnt], format="csr")
else:
    test_dense = pd.read_csv(CFG.LOAD_FEATURES_FROM)  # placeholder




## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_57/503216940.py in <cell line: 0>()
----> 1 if CFG.LOAD_FEATURES_FROM is None:
      2     test_feats1 = Paragraph_Features(test)
      3     test_feats1 = Paragraph_aggregation(test_feats1)
      4     test_feats2 = Sentence_Features(test)
      5     test_feats2 = Sentence_aggregation(test_feats2)

NameError: name 'CFG' is not defined

## === cell 23
preds = []
for i in range(15):  # same number of folds used in training
    print(f"Predicting with fold {i+1}")
    model = pickle.load(open(f"LGB_v{CFG.VER}_f{i}.pkl", "rb"))
    pred = model.predict(X_test_sparse) + a
    preds.append(pred)

pred_mean = np.mean(preds, axis=0)




## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_57/3624753249.py in <cell line: 0>()
      2 for i in range(15):  # same number of folds used in training
      3     print(f"Predicting with fold {i+1}")
----> 4     model = pickle.load(open(f"LGB_v{CFG.VER}_f{i}.pkl", "rb"))
      5     pred = model.predict(X_test_sparse) + a
      6     preds.append(pred)

NameError: name 'pickle' is not defined

## === cell 24
submission = pd.DataFrame(
    {
        "essay_id": df_test["essay_id"].values,
        "score": pred_mean.clip(1, 6).round().astype(int),
    }
)
submission.to_csv("submission.csv", index=False)
print("Submission saved. Shape:", submission.shape)
submission.head()

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_57/938746670.py in <cell line: 0>()
----> 1 submission = pd.DataFrame(
      2     {
      3         "essay_id": df_test["essay_id"].values,
      4         "score": pred_mean.clip(1, 6).round().astype(int),
      5     }

NameError: name 'pd' is not defined
