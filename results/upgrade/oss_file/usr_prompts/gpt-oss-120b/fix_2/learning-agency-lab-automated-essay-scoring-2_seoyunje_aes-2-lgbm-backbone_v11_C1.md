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
class CFG:
    SEED = 2024
    VER = 1
    LOAD_MODELS_FROM = None
    LOAD_FEATURES_FROM = None
    BASE_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/"




## === cell 1
Clean = True


def clean_memory():
    if Clean:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        gc.collect()


clean_memory()




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/3543351314.py in <cell line: 0>()
      8 
      9 
---> 10 clean_memory()
     11 
     12 

/tmp/ipykernel_56/3543351314.py in clean_memory()
      4 def clean_memory():
      5     if Clean:
----> 6         ctypes.CDLL("libc.so.6").malloc_trim(0)
      7         gc.collect()
      8 

NameError: name 'ctypes' is not defined

## === cell 2
def seed_everything():  # To produce similar result in each run
    random.seed(CFG.SEED)
    np.random.seed(CFG.SEED)
    os.environ["PYTHONHASHSEED"] = str(CFG.SEED)


seed_everything()



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/2238474941.py in <cell line: 0>()
      5 
      6 
----> 7 seed_everything()
      8 

/tmp/ipykernel_56/2238474941.py in seed_everything()
      1 def seed_everything():  # To produce similar result in each run
----> 2     random.seed(CFG.SEED)
      3     np.random.seed(CFG.SEED)
      4     os.environ["PYTHONHASHSEED"] = str(CFG.SEED)
      5 

NameError: name 'random' is not defined

## === cell 3
df_train = pd.read_csv(CFG.BASE_PATH + "train.csv")
df_train = df_train.sort_values(by="essay_id")

print("Shape of Train: ", df_train.shape)
print(df_train.head())



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/427773980.py in <cell line: 0>()
----> 1 df_train = pd.read_csv(CFG.BASE_PATH + "train.csv")
      2 df_train = df_train.sort_values(by="essay_id")
      3 
      4 print("Shape of Train: ", df_train.shape)
      5 print(df_train.head())

NameError: name 'pd' is not defined

## === cell 4
plt.figure(figsize=(12, 6))
sns.countplot(x=df_train["score"])
plt.title("Distribution of Score")
plt.xlabel("Score of Essay")
plt.ylabel("Frequency")
plt.show()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/3275623291.py in <cell line: 0>()
----> 1 plt.figure(figsize=(12, 6))
      2 sns.countplot(x=df_train["score"])
      3 plt.title("Distribution of Score")
      4 plt.xlabel("Score of Essay")
      5 plt.ylabel("Frequency")

NameError: name 'plt' is not defined

## === cell 5
df_test = pd.read_csv(CFG.BASE_PATH + "test.csv")
df_test = df_test.sort_values(by="essay_id")

print("Shape of Test: ", df_test.shape)
print(df_test.head())



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/1745607759.py in <cell line: 0>()
----> 1 df_test = pd.read_csv(CFG.BASE_PATH + "test.csv")
      2 df_test = df_test.sort_values(by="essay_id")
      3 
      4 print("Shape of Test: ", df_test.shape)
      5 print(df_test.head())

NameError: name 'pd' is not defined

## === cell 6
train = pl.from_pandas(df_train).with_columns(
    pl.col("full_text").str.split(by="\n\n").alias("paragraph")
)
test = pl.from_pandas(df_test).with_columns(
    pl.col("full_text").str.split(by="\n\n").alias("paragraph")
)

schema_train = train.schema  # MetaData
schema_test = test.schema  # MetaData



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/370250737.py in <cell line: 0>()
----> 1 train = pl.from_pandas(df_train).with_columns(
      2     pl.col("full_text").str.split(by="\n\n").alias("paragraph")
      3 )
      4 test = pl.from_pandas(df_test).with_columns(
      5     pl.col("full_text").str.split(by="\n\n").alias("paragraph")

NameError: name 'pl' is not defined

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



## === cell 8
c_re = re.compile("(%s)" % "|".join(cList.keys()))


def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]

    return c_re.sub(replace, text)


def removeHTML(x):
    html = re.compile(r"<.*?>")
    return html.sub(r"", x)  # html -> ''


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
    x = re.sub(r'[^\w\s.,;:""' "?!]", "", x)
    x = x.strip()
    return x




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/3269477440.py in <cell line: 0>()
----> 1 c_re = re.compile("(%s)" % "|".join(cList.keys()))
      2 
      3 
      4 def expandContractions(text, c_re=c_re):
      5     def replace(match):

NameError: name 're' is not defined

## === cell 9
try:
    from spellchecker import SpellChecker

    spell = SpellChecker()

    def count_misspelled_words(text: str) -> int:
        misspelled = spell.unknown(text.split())
        return len(misspelled)

except Exception:
    def count_misspelled_words(text: str) -> int:
        return 0




## === cell 10
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
    print("Caculate the length of each paragraph")
    x = x.with_columns(
        pl.col("paragraph").map_elements(lambda x: len(x)).alias("paragraph_len")
    )
    x = x.with_columns(
        pl.col("paragraph")
        .map_elements(lambda x: count_misspelled_words(x))
        .alias("paragraph_misspelled_cnt")
    )
    x = x.with_columns(
        pl.col("paragraph")
        .map_elements(lambda x: x.count(","))
        .alias("paragraph_comma_cnt")
    )
    print("Caculate the number of sentences and words in each paragraph")
    x = x.with_columns(
        pl.col("paragraph")
        .map_elements(lambda x: len(x.split(".")))
        .alias("paragraph_sentence_cnt"),
        pl.col("paragraph")
        .map_elements(lambda x: len(x.split(" ")))
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
        *[
            pl.col("paragraph")
            .filter((pl.col("paragraph_len") <= 300) & (pl.col("paragraph_len") > 100))
            .count()
            .alias(f"short_paragraph_cnt")
        ],
        *[
            pl.col("paragraph")
            .filter((pl.col("paragraph_len") <= 500) & (pl.col("paragraph_len") > 300))
            .count()
            .alias(f"mid_paragraph_cnt")
        ],
        *[
            pl.col("paragraph")
            .filter((pl.col("paragraph_len") <= 700) & (pl.col("paragraph_len") > 500))
            .count()
            .alias(f"long_paragraph_cnt")
        ],
        *[
            pl.col("paragraph")
            .filter(pl.col("paragraph_sentence_cnt") >= i)
            .count()
            .alias(f"paragraph_sentence_{i}_cnt")
            for i in [2, 4, 6, 8, 10]
        ],
        *[
            pl.col("paragraph")
            .filter(
                (pl.col("paragraph_sentence_cnt") <= 4)
                & (pl.col("paragraph_sentence_cnt") > 2)
            )
            .count()
            .alias(f"short_paragraph_sentence_cnt")
        ],
        *[
            pl.col("paragraph")
            .filter(
                (pl.col("paragraph_sentence_cnt") <= 8)
                & (pl.col("paragraph_sentence_cnt") > 4)
            )
            .count()
            .alias(f"mid_paragraph_sentence_cnt")
        ],
        *[
            pl.col("paragraph")
            .filter(
                (pl.col("paragraph_sentence_cnt") <= 10)
                & (pl.col("paragraph_sentence_cnt") > 8)
            )
            .count()
            .alias(f"long_paragraph_sentence_cnt")
        ],
        *[
            pl.col("paragraph")
            .filter(pl.col("paragraph_word_cnt") >= i)
            .count()
            .alias(f"paragraph_word_{i}_cnt")
            for i in [20, 40, 60, 90, 120]
        ],
        *[
            pl.col("paragraph")
            .filter(
                (pl.col("paragraph_word_cnt") <= 40)
                & (pl.col("paragraph_word_cnt") > 20)
            )
            .count()
            .alias(f"short_paragraph_word_cnt")
        ],
        *[
            pl.col("paragraph")
            .filter(
                (pl.col("paragraph_word_cnt") <= 90)
                & (pl.col("paragraph_word_cnt") > 40)
            )
            .count()
            .alias(f"mid_paragraph_word_cnt")
        ],
        *[
            pl.col("paragraph")
            .filter(
                (pl.col("paragraph_word_cnt") <= 120)
                & (pl.col("paragraph_word_cnt") > 90)
            )
            .count()
            .alias(f"long_paragraph_word_cnt")
        ],
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
        *[
            pl.col("paragraph")
            .filter(
                (pl.col("paragraph_misspelled_cnt") <= 8)
                & (pl.col("paragraph_misspelled_cnt") > 4)
            )
            .count()
            .alias(f"short_paragraph_misspelled_cnt")
        ],
        *[
            pl.col("paragraph")
            .filter(
                (pl.col("paragraph_misspelled_cnt") <= 12)
                & (pl.col("paragraph_misspelled_cnt") > 8)
            )
            .count()
            .alias(f"mid_paragraph_misspelled_cnt")
        ],
        *[
            pl.col("paragraph")
            .filter(
                (pl.col("paragraph_misspelled_cnt") <= 16)
                & (pl.col("paragraph_misspelled_cnt") > 12)
            )
            .count()
            .alias(f"long_paragraph_misspelled_cnt")
        ],
        *[pl.col("paragraph").count().alias("paragraph_cnt")],
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
    df = df.to_pandas()  # polars -> pandas

    return df




## === cell 11
sentence_features = ["sentence_len", "sentence_word_cnt", "sentence_misspelled_cnt"]


def Sentence_Features(x):
    print("Preprocess full_text and use periods to segment sentences in the text")
    x = x.with_columns(
        pl.col("full_text")
        .map_elements(lambda x: dataPreprocessing(x))
        .str.split(".")
        .alias("sentence")
    )
    x = x.explode("sentence")

    print("Caculate the length of a sentence")
    x = x.with_columns(
        pl.col("sentence").map_elements(lambda x: len(x)).alias("sentence_len")
    )
    x = x.filter(pl.col("sentence_len") > 3)
    x = x.with_columns(
        pl.col("sentence")
        .map_elements(lambda x: count_misspelled_words(x))
        .alias("sentence_misspelled_cnt")
    )
    x = x.with_columns(
        pl.col("sentence")
        .map_elements(lambda x: len(x.replace(" ", "")))
        .alias("only_sentence_len")
    )
    print("Count the number of words in each sentence")
    x = x.with_columns(
        pl.col("sentence")
        .map_elements(lambda x: len(x.split(" ")))
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
        *[
            pl.col("sentence")
            .filter((pl.col("sentence_len") <= 70) & (pl.col("sentence_len") > 40))
            .count()
            .alias(f"short_sentence_cnt")
        ],
        *[
            pl.col("sentence")
            .filter((pl.col("sentence_len") <= 100) & (pl.col("sentence_len") > 70))
            .count()
            .alias(f"mid_sentence_cnt")
        ],
        *[
            pl.col("sentence")
            .filter((pl.col("sentence_len") <= 140) & (pl.col("sentence_len") > 100))
            .count()
            .alias(f"long_sentence_cnt")
        ],
        *[
            pl.col("sentence")
            .filter(pl.col("only_sentence_len") >= i)
            .count()
            .alias(f"only_sentence_{i}_cnt")
            for i in [40, 60, 80, 100, 120]
        ],
        *[
            pl.col("sentence")
            .filter(
                (pl.col("only_sentence_len") <= 60) & (pl.col("only_sentence_len") > 40)
            )
            .count()
            .alias(f"short_only_sentence_cnt")
        ],
        *[
            pl.col("sentence")
            .filter(
                (pl.col("only_sentence_len") <= 100)
                & (pl.col("only_sentence_len") > 60)
            )
            .count()
            .alias(f"mid_only_sentence_cnt")
        ],
        *[
            pl.col("sentence")
            .filter(
                (pl.col("only_sentence_len") <= 120)
                & (pl.col("only_sentence_len") > 100)
            )
            .count()
            .alias(f"long_only_sentence_cnt")
        ],
        *[
            pl.col("sentence")
            .filter(pl.col("sentence_word_cnt") >= i)
            .count()
            .alias(f"sentence_word_{i}_cnt")
            for i in [10, 15, 20, 25]
        ],
        *[
            pl.col("sentence")
            .filter(
                (pl.col("sentence_word_cnt") <= 15) & (pl.col("sentence_word_cnt") > 10)
            )
            .count()
            .alias(f"short_sentence_word_cnt")
        ],
        *[
            pl.col("sentence")
            .filter(
                (pl.col("sentence_word_cnt") <= 20) & (pl.col("sentence_word_cnt") > 15)
            )
            .count()
            .alias(f"mid_sentence_word_cnt")
        ],
        *[
            pl.col("sentence")
            .filter(
                (pl.col("sentence_word_cnt") <= 25) & (pl.col("sentence_word_cnt") > 20)
            )
            .count()
            .alias(f"long_sentence_word_cnt")
        ],
        *[pl.col("sentence").count().alias("sentence_cnt")],
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
        *[
            (pl.col("short_sentence_cnt") / pl.col("sentence_cnt")).alias(
                f"short_sentence_cnt_ratio"
            )
        ],
        *[
            (pl.col("mid_sentence_cnt") / pl.col("sentence_cnt")).alias(
                f"mid_sentence_cnt_ratio"
            )
        ],
        *[
            (pl.col("long_sentence_cnt") / pl.col("sentence_cnt")).alias(
                f"long_sentence_cnt_ratio"
            )
        ],
    ).sort("essay_id")

    df = df.to_pandas()  # polars -> pandas

    return df




## === cell 12
word_features = [
    "word_len",
]


def Word_Features(x):
    print("Preprocess full_text and use spaces to seperate words fro the text")

    x = x.with_columns(
        pl.col("full_text")
        .map_elements(lambda x: dataPreprocessing(x))
        .str.split(" ")
        .alias("word")
    )
    x = x.explode("word")

    print("Caculate the length of a word")
    x = x.with_columns(pl.col("word").map_elements(lambda x: len(x)).alias("word_len"))
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
        *[
            pl.col("word")
            .filter((pl.col("word_len") <= 4) & (pl.col("word_len") > 2))
            .count()
            .alias(f"short_word_cnt")
        ],
        *[
            pl.col("word")
            .filter((pl.col("word_len") <= 6) & (pl.col("word_len") > 4))
            .count()
            .alias(f"mid_word_cnt")
        ],
        *[
            pl.col("word")
            .filter((pl.col("word_len") <= 10) & (pl.col("word_len") > 6))
            .count()
            .alias(f"long_word_cnt")
        ],
        *[pl.col("word").count().alias("word_cnt")],
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
            (pl.col(f"short_word_cnt") / pl.col(f"word_{i}_cnt_v2")).alias(
                f"short_word_ratio_{i}"
            )
            for i in [1, 2, 3]
        ],
        *[
            (pl.col(f"mid_word_cnt") / pl.col(f"word_{i}_cnt_v2")).alias(
                f"mid_word_ratio_{i}"
            )
            for i in [1, 2, 3]
        ],
        *[
            (pl.col(f"long_word_cnt") / pl.col(f"word_{i}_cnt_v2")).alias(
                f"long_word_ratio_{i}"
            )
            for i in [1, 2, 3]
        ],
    ).sort("essay_id")

    df = df.to_pandas()  # polars -> pandas

    return df




## === cell 13
vectorizer = TfidfVectorizer(
    tokenizer=lambda x: x,
    preprocessor=lambda x: x,
    token_pattern=None,
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(1, 4),
    min_df=0.05,
    max_df=0.95,
    sublinear_tf=True,  # Term Frequency Log Scaling
)

train_a = train.with_columns(
    pl.col("full_text").map_elements(lambda x: dataPreprocessing(x))
)
train_tfid = vectorizer.fit_transform([i for i in train_a["full_text"]])

dense_matrix = train_tfid.toarray()

df = pd.DataFrame(dense_matrix)
df.columns = [f"tfidf_{i}" for i in range(len(df.columns))]
df["essay_id"] = df_train["essay_id"]



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/511798754.py in <cell line: 0>()
----> 1 vectorizer = TfidfVectorizer(
      2     tokenizer=lambda x: x,
      3     preprocessor=lambda x: x,
      4     token_pattern=None,
      5     strip_accents="unicode",

NameError: name 'TfidfVectorizer' is not defined

## === cell 14
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


train_b = train.with_columns(
    pl.col("full_text").map_elements(lambda x: dataPreprocessing(x))
)
train_cnt = vectorizer_cnt.fit_transform([i for i in train_b["full_text"]])

dense_matrix2 = train_cnt.toarray()

df2 = pd.DataFrame(dense_matrix2)
df2.columns = [f"cnt_{i}" for i in range(len(df2.columns))]
df2["essay_id"] = df_train["essay_id"]



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/2996382268.py in <cell line: 0>()
----> 1 vectorizer_cnt = CountVectorizer(
      2     tokenizer=lambda x: x,
      3     preprocessor=lambda x: x,
      4     token_pattern=None,
      5     strip_accents="unicode",

NameError: name 'CountVectorizer' is not defined

## === cell 15
if CFG.LOAD_FEATURES_FROM is None:

    train_feats1 = Paragraph_Features(train)
    train_feats1 = Paragraph_aggregation(train_feats1)
    train_feats2 = Sentence_Features(train)
    train_feats2 = Sentence_aggregation(train_feats2)
    train_feats3 = Word_Features(train)
    train_feats3 = Word_aggregation(train_feats3)

    train_feats = train_feats1.merge(train_feats2, on="essay_id", how="left")
    train_feats = train_feats.merge(train_feats3, on="essay_id", how="left")
    train_feats = train_feats.merge(df, on="essay_id", how="left")
    train_feats = train_feats.merge(df2, on="essay_id", how="left")
    train_feats["score"] = df_train["score"].values
else:
    train_feats = pd.read_csv(CFG.LOAD_FEATURES_FROM)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/2623440287.py in <cell line: 0>()
      1 if CFG.LOAD_FEATURES_FROM is None:
      2 
----> 3     train_feats1 = Paragraph_Features(train)
      4     train_feats1 = Paragraph_aggregation(train_feats1)
      5     train_feats2 = Sentence_Features(train)

NameError: name 'train' is not defined

## === cell 16
if CFG.LOAD_FEATURES_FROM is None:
    print("Save train_feats.csv")
    train_feats.to_csv(f"train_feats_{CFG.VER}.csv", index=False)
else:
    print("Load train_feats.csv")
    train_feats = pd.read_csv(CFG.LOAD_FEATURES_FROM)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/152654945.py in <cell line: 0>()
      1 if CFG.LOAD_FEATURES_FROM is None:
      2     print("Save train_feats.csv")
----> 3     train_feats.to_csv(f"train_feats_{CFG.VER}.csv", index=False)
      4 else:
      5     print("Load train_feats.csv")

NameError: name 'train_feats' is not defined

## === cell 17
print(train_feats.head())



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/2022479248.py in <cell line: 0>()
----> 1 print(train_feats.head())
      2 

NameError: name 'train_feats' is not defined

## === cell 18
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import cohen_kappa_score

import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

print("LightGBM Version: ", lgb.__version__)




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
    f = 1 / 2 * np.sum((preds - labels) ** 2)
    g = 1 / 2 * np.sum((preds - a) ** 2 + b)
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




## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/75358094.py in <cell line: 0>()
----> 1 categorical_columns = train_feats.select_dtypes(
      2     include=["object", "category"]
      3 ).columns.tolist()
      4 FEATURES = [
      5     col for col in train_feats.columns if col not in categorical_columns + ["score"]

NameError: name 'train_feats' is not defined

## === cell 21
def lightgbm():
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
            metrics="None",
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

        oof = model.predict(valid_x, num_iteration=model.best_iteration_)
        all_oof.append(oof + a)
        all_true.append(valid_y.values + a)

        del train_x, train_y, valid_x, valid_y, oof, model
        clean_memory()

    all_oof = np.concatenate(all_oof)
    all_true = np.concatenate(all_true)

    cv = cohen_kappa_score(
        all_true, np.clip(all_oof, 1, 6).round(), weights="quadratic"
    )
    print("CV Score for LightGBM = ", cv)
    cm = confusion_matrix(
        all_true, np.clip(all_oof, 1, 6).round(), labels=[x for x in range(1, 7)]
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=[x for x in range(1, 7)]
    )
    disp.plot()
    plt.show()




## === cell 22
if CFG.LOAD_MODELS_FROM is None:
    print("Training LightGBM")
    lightgbm()
else:
    pass



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/3316394020.py in <cell line: 0>()
      1 if CFG.LOAD_MODELS_FROM is None:
      2     print("Training LightGBM")
----> 3     lightgbm()
      4 else:
      5     pass

/tmp/ipykernel_56/927601398.py in lightgbm()
      5     skf = StratifiedKFold(n_splits=15, random_state=CFG.SEED, shuffle=True)
      6     for i, (train_index, valid_index) in enumerate(
----> 7         skf.split(train_feats, train_feats[TARGET])
      8     ):
      9 

NameError: name 'train_feats' is not defined

## === cell 23
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



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/4215605823.py in <cell line: 0>()
      1 # Load the first fold model (it exists after training) to show feature importance
----> 2 model = pickle.load(open(f"LGB_v{CFG.VER}_f0.pkl", "rb"))
      3 
      4 df_importance = pd.DataFrame(
      5     {

NameError: name 'pickle' is not defined

## === cell 24
test_a = test.with_columns(
    pl.col("full_text").map_elements(lambda x: dataPreprocessing(x))
)
test_tfid = vectorizer.transform([i for i in test_a["full_text"]])
dense_matrix = test_tfid.toarray()
df3 = pd.DataFrame(dense_matrix)
tfid_columns = [f"tfidf_{i}" for i in range(len(df3.columns))]
df3.columns = tfid_columns
df3["essay_id"] = df_test["essay_id"]



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/231055096.py in <cell line: 0>()
----> 1 test_a = test.with_columns(
      2     pl.col("full_text").map_elements(lambda x: dataPreprocessing(x))
      3 )
      4 test_tfid = vectorizer.transform([i for i in test_a["full_text"]])
      5 dense_matrix = test_tfid.toarray()

NameError: name 'test' is not defined

## === cell 25
test_b = test.with_columns(
    pl.col("full_text").map_elements(lambda x: dataPreprocessing(x))
)
test_cnt = vectorizer_cnt.transform([i for i in test_b["full_text"]])
dense_matrix = test_cnt.toarray()
df4 = pd.DataFrame(dense_matrix)
cnt_columns = [f"cnt_{i}" for i in range(len(df4.columns))]
df4.columns = cnt_columns
df4["essay_id"] = df_test["essay_id"]



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/3160514794.py in <cell line: 0>()
----> 1 test_b = test.with_columns(
      2     pl.col("full_text").map_elements(lambda x: dataPreprocessing(x))
      3 )
      4 test_cnt = vectorizer_cnt.transform([i for i in test_b["full_text"]])
      5 dense_matrix = test_cnt.toarray()

NameError: name 'test' is not defined

## === cell 26
test_feats1 = Paragraph_Features(test)
test_feats1 = Paragraph_aggregation(test_feats1)
test_feats2 = Sentence_Features(test)
test_feats2 = Sentence_aggregation(test_feats2)
test_feats3 = Word_Features(test)
test_feats3 = Word_aggregation(test_feats3)



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/2387858613.py in <cell line: 0>()
----> 1 test_feats1 = Paragraph_Features(test)
      2 test_feats1 = Paragraph_aggregation(test_feats1)
      3 test_feats2 = Sentence_Features(test)
      4 test_feats2 = Sentence_aggregation(test_feats2)
      5 test_feats3 = Word_Features(test)

NameError: name 'test' is not defined

## === cell 27
test_feats = test_feats1.merge(test_feats2, on="essay_id", how="left")
test_feats = test_feats.merge(test_feats3, on="essay_id", how="left")
test_feats = test_feats.merge(df3, on="essay_id", how="left")
test_feats = test_feats.merge(df4, on="essay_id", how="left")
print("Shape of test_feats:", test_feats.shape)
display(test_feats.head())



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/14690776.py in <cell line: 0>()
----> 1 test_feats = test_feats1.merge(test_feats2, on="essay_id", how="left")
      2 test_feats = test_feats.merge(test_feats3, on="essay_id", how="left")
      3 test_feats = test_feats.merge(df3, on="essay_id", how="left")
      4 test_feats = test_feats.merge(df4, on="essay_id", how="left")
      5 print("Shape of test_feats:", test_feats.shape)

NameError: name 'test_feats1' is not defined

## === cell 28
preds = []
categorical_columns = test_feats.select_dtypes(
    include=["object", "category"]
).columns.tolist()
FEATURES = [col for col in test_feats.columns if col not in categorical_columns]

for i in [0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 13, 14]:
    print(f"Fold {i+1}")
    model = pickle.load(open(f"LGB_v{CFG.VER}_f{i}.pkl", "rb"))
    pred = model.predict(test_feats[FEATURES]) + a
    preds.append(pred)

pred1 = np.mean(preds, axis=0)



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/2576228775.py in <cell line: 0>()
      1 preds = []
----> 2 categorical_columns = test_feats.select_dtypes(
      3     include=["object", "category"]
      4 ).columns.tolist()
      5 FEATURES = [col for col in test_feats.columns if col not in categorical_columns]

NameError: name 'test_feats' is not defined

## === cell 29
sub = pd.DataFrame({"essay_id": df_test.essay_id.values})
sub["score"] = pred1.clip(1, 6).round()
sub.to_csv("submission.csv", index=False)
print("Submission shape", sub.shape)
sub.head()

## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/1533121587.py in <cell line: 0>()
----> 1 sub = pd.DataFrame({"essay_id": df_test.essay_id.values})
      2 sub["score"] = pred1.clip(1, 6).round()
      3 sub.to_csv("submission.csv", index=False)
      4 print("Submission shape", sub.shape)
      5 sub.head()

NameError: name 'pd' is not defined
