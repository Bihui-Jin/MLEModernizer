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
Predict the word or phrase from tweets that exemplifies the labelled sentiment.

## Metric
Word-level Jaccard score.

## Submission Format
For each ID in the test set, you must predict the string that best supports the sentiment for the tweet in question. Note that the selected text _needs_ to be **quoted** and **complete** (include punctuation, etc. - the above code splits ONLY on whitespace) to work correctly. The file should contain a header and have the following format:
```
textID,selected_text
2,"very good"
5,"I don't care"
6,"bad"
8,"it was, yes"
etc.
```

## Dataset
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

- `textID` - unique ID for each piece of text
- `text` - the text of the tweet
- `sentiment` - the general sentiment of the tweet
- `selected_text` - [train only] the text that supports the tweet's sentiment

# 2. Python version

3.8

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
plotly==5.24.1
plotly-express==0.4.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
spacy==3.8.7
spacy-legacy==3.0.12
spacy-loggers==1.0.5
tqdm==4.67.1
wordcloud==1.9.4

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (130 lines)
            sample_submission.csv (2750 lines)
            sample_submission.csv.zip (18.6 kB)
            test.csv (2750 lines)
            test.csv.zip (114.6 kB)
            train.csv (24733 lines)
            train.csv.zip (1.2 MB)
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
        input/
            description.md (130 lines)
            sample_submission.csv (2750 lines)
            sample_submission.csv.zip (18.6 kB)
            test.csv (2750 lines)
            test.csv.zip (114.6 kB)
            train.csv (24733 lines)
            train.csv.zip (1.2 MB)
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
        working/
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
```

-> data/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> data/test.csv has 2749 rows and 3 columns.
The columns are: textID, text, sentiment

-> data/train.csv has 24732 rows and 4 columns.
The columns are: textID, text, selected_text, sentiment

-> data/tweet-sentiment-extraction/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> data/tweet-sentiment-extraction/test.csv has 2749 rows and 3 columns.
The columns are: textID, text, sentiment

-> data/tweet-sentiment-extraction/train.csv has 24732 rows and 4 columns.
The columns are: textID, text, selected_text, sentiment

-> input/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> (stopped after 10 files for performance)

# 5. Target score

0.6507

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import re
import nltk
import string
import random

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

from tqdm import tqdm
import spacy
from spacy.util import compounding
from spacy.util import minibatch


for pkg in [
    "punkt",
    "stopwords",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
]:
    try:
        nltk.data.find(pkg)
    except LookupError:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass



## === cell 1
train_data = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
test_data = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
submission_data = pd.read_csv(
    "/kaggle/input/tweet-sentiment-extraction/sample_submission.csv"
)



## === cell 2
print(train_data.shape, test_data.shape, submission_data.shape)
print(train_data.head(2))
print(test_data.head(2))
print(submission_data.head(2))



## === cell 3
train_data.isna().any()



## === cell 4
test_data.isna().any()



## === cell 5
train_data.loc[train_data["selected_text"].isnull()].head()



## === cell 6
train_data.loc[train_data["text"].isnull()].head()



## === cell 7
train_data_new = train_data.drop([314]).reset_index(drop=True)



## === cell 8
train_data_new.isna().any()



## === cell 9
train_data_new["sentiment"].unique()




## === cell 10
def count_senti(df):
    s = df["sentiment"].value_counts()
    percent = df["sentiment"].value_counts(normalize=True)
    return pd.concat([s, percent], axis=1, keys=["Sum", "Percent"])




## === cell 11
print("Sentiments for train data")
senti_train = count_senti(train_data_new)
print(senti_train)
print("Sentiments for test data")
senti_test = count_senti(test_data)
print(senti_test)



## === cell 13
df_train = train_data_new.copy()
df_test = test_data.copy()



## === cell 14
df_train[df_train["sentiment"] == "positive"].head()



## === cell 15
df_train[df_train["sentiment"] == "neutral"].head()



## === cell 16
df_train[df_train["sentiment"] == "negative"].head()




## === cell 17
def text_cleaning(txt):
    """
    Convert given text to lower case, remove all non-word characters, digits, links.
    """
    txt = str(txt).lower()
    txt = re.sub(r"https?://\S+|www\.\S+", "", txt)
    txt = re.sub(r"\[.*?\]", "", txt)
    txt = re.sub(r"<.*?>+", "", txt)
    txt = re.sub(r"[%s]" % re.escape(string.punctuation), " ", txt)
    txt = re.sub(r"\n", "", txt)
    txt = re.sub(r"\w*\d\w*", "", txt)
    return txt




## === cell 18
def rm_stopword(text):
    text_tokens = word_tokenize(text)
    stop_list = stopwords.words("english")
    new_text = [word for word in text_tokens if word not in stop_list]
    final_text = " ".join(new_text)
    return final_text




## === cell 19
df_train["clean_text"] = df_train["text"].apply(lambda x: text_cleaning(x))
df_train["clean_selected_text"] = df_train["selected_text"].apply(
    lambda x: text_cleaning(x)
)



## === cell 20
df_train.head(5)



## === cell 21
df_train["clean_text"] = df_train["clean_text"].apply(lambda x: rm_stopword(x))
df_train["clean_selected_text"] = df_train["clean_selected_text"].apply(
    lambda x: rm_stopword(x)
)



## === cell 22
df_train.head(5)




## === cell 23
def count_words(df, feature, senti):
    word_list = []
    for x in df[df["sentiment"] == senti][feature].astype(str).str.split():
        for i in x:
            word_list.append(i)
    cnt = Counter()
    for word in word_list:
        cnt[word] += 1
    df_cnt = pd.DataFrame(cnt.most_common(10), columns=["Freq_words", "Freq"])
    return df_cnt




## === cell 24
positive_top10 = count_words(df_train, "clean_text", "positive")
print(positive_top10)



## === cell 25
neutral_top10 = count_words(df_train, "clean_text", "neutral")
print(neutral_top10)



## === cell 26
negative_top10 = count_words(df_train, "clean_text", "negative")
print(negative_top10)




## === cell 27
def pos_freq(df, feature, senti):
    total_pos_count = []
    for x in df[df["sentiment"] == senti][feature].astype(str).str.split():
        pos_count = nltk.pos_tag(x)
        total_pos_count.extend(pos_count)
    tag_freq = nltk.FreqDist(tag for (_, tag) in total_pos_count)
    ans = tag_freq.most_common()[0:10]
    return ans




## === cell 28
print(pos_freq(df_train, "clean_text", "negative")[:5])



## === cell 29
print(pos_freq(df_train, "text", "negative")[:5])



## === cell 30
print(pos_freq(df_train, "text", "positive")[:5])



## === cell 31
train_df1 = train_data_new.copy()
test_df1 = test_data.copy()
submission_df1 = submission_data.copy()



## === cell 32
train_df1["Num_words_text"] = train_df1["text"].apply(lambda x: len(str(x).split()))



## === cell 33
train_df1 = train_df1[train_df1["Num_words_text"] >= 3].reset_index(drop=True)




## === cell 34
def save_model(output_dir, nlp, new_model_name):
    """Save model to /kaggle/working so it can be reloaded later in the same run."""
    output_dir = os.path.join("/kaggle/working", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    nlp.meta["name"] = new_model_name
    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)




## === cell 35
def train(train_data, output_dir, n_iter=20, model=None):
    """Train a spaCy NER model (spaCy v3 compatible API)."""
    if model is not None:
        nlp = spacy.load(os.path.join("/kaggle/working", output_dir))
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")
        print("Created blank 'en' model")

    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    for _, annotations in train_data:
        for start, end, label in annotations.get("entities"):
            ner.add_label(label)

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.initialize() if model is None else nlp.resume_training()

        for itn in tqdm(range(n_iter)):
            random.shuffle(train_data)
            batches = minibatch(train_data, size=compounding(4.0, 500.0, 1.001))
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,
                    annotations,
                    drop=0.5,
                    sgd=optimizer,
                    losses=losses,
                )
            print("Losses", losses)

    save_model(output_dir, nlp, "st_ner")




## === cell 36
def get_model_out_path(sentiment):
    if sentiment == "positive":
        return "models/model_pos"
    elif sentiment == "negative":
        return "models/model_neg"
    return None




## === cell 37
def get_training_data(sentiment):
    """
    Build spaCy NER training data: (text, {"entities": [(start,end,label)]})
    FIX: skip rows where selected_text isn't found to avoid invalid offsets.
    """
    train_data = []
    df_s = train_df1[train_df1["sentiment"] == sentiment]
    for _, row in df_s.iterrows():
        selected_text = str(row.selected_text)
        text = str(row.text)

        start = text.find(selected_text)
        if start == -1:
            continue
        end = start + len(selected_text)
        if end <= start:
            continue

        train_data.append((text, {"entities": [(start, end, "selected_text")]}))
    return train_data




## === cell 38
sentiment = "positive"
train_data_pos = get_training_data(sentiment)
model_path_pos = get_model_out_path(sentiment)
train(train_data_pos, model_path_pos, n_iter=3, model=None)



## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/261136555.py in <cell line: 0>()
      2 train_data_pos = get_training_data(sentiment)
      3 model_path_pos = get_model_out_path(sentiment)
----> 4 train(train_data_pos, model_path_pos, n_iter=3, model=None)
      5 

/tmp/ipykernel_11/2576253766.py in train(train_data, output_dir, n_iter, model)
     29             for batch in batches:
     30                 texts, annotations = zip(*batch)
---> 31                 nlp.update(
     32                     texts,
     33                     annotations,

/usr/local/lib/python3.11/dist-packages/spacy/language.py in update(self, examples, _, drop, sgd, losses, component_cfg, exclude, annotates)
   1173         """
   1174         if _ is not None:
-> 1175             raise ValueError(Errors.E989)
   1176         if losses is None:
   1177             losses = {}

ValueError: [E989] `nlp.update()` was called with two positional arguments. This may be due to a backwards-incompatible change to the format of the training data in spaCy 3.0 onwards. The 'update' function should now be called with a batch of Example objects, instead of `(text, annotation)` tuples. 

## === cell 39
sentiment = "negative"
train_data_neg = get_training_data(sentiment)
model_path_neg = get_model_out_path(sentiment)
train(train_data_neg, model_path_neg, n_iter=3, model=None)




## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3776775377.py in <cell line: 0>()
      2 train_data_neg = get_training_data(sentiment)
      3 model_path_neg = get_model_out_path(sentiment)
----> 4 train(train_data_neg, model_path_neg, n_iter=3, model=None)
      5 
      6 

/tmp/ipykernel_11/2576253766.py in train(train_data, output_dir, n_iter, model)
     29             for batch in batches:
     30                 texts, annotations = zip(*batch)
---> 31                 nlp.update(
     32                     texts,
     33                     annotations,

/usr/local/lib/python3.11/dist-packages/spacy/language.py in update(self, examples, _, drop, sgd, losses, component_cfg, exclude, annotates)
   1173         """
   1174         if _ is not None:
-> 1175             raise ValueError(Errors.E989)
   1176         if losses is None:
   1177             losses = {}

ValueError: [E989] `nlp.update()` was called with two positional arguments. This may be due to a backwards-incompatible change to the format of the training data in spaCy 3.0 onwards. The 'update' function should now be called with a batch of Example objects, instead of `(text, annotation)` tuples. 

## === cell 40
def predict_entities(text, model):
    doc = model(text)
    ent_array = []
    for ent in doc.ents:
        start = text.find(ent.text)
        end = start + len(ent.text)
        new_int = [start, end, ent.label_]
        if start != -1 and end > start and new_int not in ent_array:
            ent_array.append([start, end, ent.label_])
    selected_text = (
        text[ent_array[0][0] : ent_array[0][1]] if len(ent_array) > 0 else text
    )
    return selected_text




## === cell 41
selected_texts = []
MODELS_BASE_PATH = "/kaggle/working/models/"

print("Loading Models from", MODELS_BASE_PATH)
model_pos = spacy.load(os.path.join(MODELS_BASE_PATH, "model_pos"))
model_neg = spacy.load(os.path.join(MODELS_BASE_PATH, "model_neg"))

for _, row in df_test.iterrows():
    text = str(row.text)
    if row.sentiment == "neutral" or len(text.split()) <= 2:
        selected_texts.append(text)
    elif row.sentiment == "positive":
        selected_texts.append(predict_entities(text, model_pos))
    else:
        selected_texts.append(predict_entities(text, model_neg))

test_df1["selected_text"] = selected_texts



## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/3026120343.py in <cell line: 0>()
      3 
      4 print("Loading Models from", MODELS_BASE_PATH)
----> 5 model_pos = spacy.load(os.path.join(MODELS_BASE_PATH, "model_pos"))
      6 model_neg = spacy.load(os.path.join(MODELS_BASE_PATH, "model_neg"))
      7 

/usr/local/lib/python3.11/dist-packages/spacy/__init__.py in load(name, vocab, disable, enable, exclude, config)
     50     RETURNS (Language): The loaded nlp object.
     51     """
---> 52     return util.load_model(
     53         name,
     54         vocab=vocab,

/usr/local/lib/python3.11/dist-packages/spacy/util.py in load_model(name, vocab, disable, enable, exclude, config)
    482     if name in OLD_MODEL_SHORTCUTS:
    483         raise IOError(Errors.E941.format(name=name, full=OLD_MODEL_SHORTCUTS[name]))  # type: ignore[index]
--> 484     raise IOError(Errors.E050.format(name=name))
    485 
    486 

OSError: [E050] Can't find model '/kaggle/working/models/model_pos'. It doesn't seem to be a Python package or a valid path to a data directory.

## === cell 42
submission_df1 = submission_df1.copy()
submission_df1["selected_text"] = test_df1["selected_text"].values
submission_path = "submission.csv"
submission_df1[["textID", "selected_text"]].to_csv(submission_path, index=False)
print(submission_df1.head(10))
print(
    "Wrote:",
    submission_path,
    "with shape",
    submission_df1[["textID", "selected_text"]].shape,
)

## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'selected_text'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1672583546.py in <cell line: 0>()
      1 # Ensure required submission format: columns ['textID','selected_text'] and csv suffix.
      2 submission_df1 = submission_df1.copy()
----> 3 submission_df1["selected_text"] = test_df1["selected_text"].values
      4 submission_path = "submission.csv"
      5 submission_df1[["textID", "selected_text"]].to_csv(submission_path, index=False)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'selected_text'
