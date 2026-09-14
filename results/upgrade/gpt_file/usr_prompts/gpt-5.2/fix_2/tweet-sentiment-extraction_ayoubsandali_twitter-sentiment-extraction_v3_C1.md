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

3.9

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
seaborn==0.12.2
sklearn-pandas==2.2.0
spacy==3.8.7
spacy-legacy==3.0.12
spacy-loggers==1.0.5
tqdm==4.67.1

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

0.65537

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import re
import warnings

import numpy as np
import pandas as pd
import spacy
from spacy.util import compounding, minibatch
from tqdm import tqdm

warnings.filterwarnings("ignore")

train_x = pd.read_csv("../input/tweet-sentiment-extraction/train.csv")
test_x = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")
train_x.dropna(inplace=True)
train_x.head()



## === cell 1
train_x.describe()




## === cell 2
def number_words(text):
    text = re.sub(r"[^\w\s]", "", str(text))
    text = text.strip()
    text_list = text.split()
    return len(text_list)


def jaccard(str1, str2):
    a = set(str(str1).lower().split())
    b = set(str(str2).lower().split())
    c = a.intersection(b)
    denom = len(a) + len(b) - len(c)
    return float(len(c)) / denom if denom > 0 else 0.0


df = train_x.assign(jaccard_score=np.nan)
df["jaccard_score"] = [
    jaccard(df.at[i, "selected_text"], df.at[i, "text"]) for i in df.index
]
df["St_words_number"] = [number_words(df.at[i, "selected_text"]) for i in df.index]
df["text_word_number"] = [number_words(df.at[i, "text"]) for i in df.index]
df["diff_number_words"] = df["text_word_number"] - df["St_words_number"]
df.head()



## === cell 3
import matplotlib.pyplot as plt
import seaborn as sns

pd.plotting.register_matplotlib_converters()

plt.figure(figsize=(12, 6))
sns.kdeplot(data=df["text_word_number"], fill=True)
sns.kdeplot(data=df["St_words_number"], fill=True)
plt.close()



## === cell 4
plt.figure(figsize=(12, 6))
df_neutral = df[df["sentiment"] == "neutral"]
sns.histplot(df_neutral["jaccard_score"], kde=False)
plt.close()



## === cell 5
plt.figure(figsize=(12, 6))
df_positive = df[df["sentiment"] == "positive"]
sns.kdeplot(data=df_positive["jaccard_score"], label="positive", fill=True)
plt.close()



## === cell 6
plt.figure(figsize=(12, 6))
df_negative = df[df["sentiment"] == "negative"]
sns.kdeplot(data=df_negative["jaccard_score"], label="negative", color="red", fill=True)
plt.close()



## === cell 7
k = df[df["text_word_number"] <= 3]
k.groupby("sentiment")["jaccard_score"].mean()



## === cell 8
k = df[df["text_word_number"] <= 2]
k.groupby("sentiment")["jaccard_score"].mean()



## === cell 9
import string

df["text"] = df["text"].astype(str).str.replace(r"[^\w\s]", "", regex=True)
df["selected_text"] = (
    df["selected_text"].astype(str).str.replace(r"[^\w\s]", "", regex=True)
)
k = df.loc[(df["text_word_number"] <= 2) & (df["jaccard_score"] < 1)]
k.head()



## === cell 10
df_train = pd.read_csv("../input/tweet-sentiment-extraction/train.csv")
df_test = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")
df_submission = pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")



## === cell 11
df_train["Num_words_text"] = df_train["text"].apply(lambda x: len(str(x).split()))
df_train = df_train[df_train["Num_words_text"] > 3].reset_index(drop=True)




## === cell 12
def save_model(output_dir, nlp, new_model_name):
    """Save model under ./tse-spacy-model/{output_dir} (relative to working dir)."""
    output_dir = f"./tse-spacy-model/{output_dir}"
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        nlp.meta["name"] = new_model_name
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)




## === cell 13
def get_model_out_path(sentiment):
    """Returns model output path (relative inside ./tse-spacy-model/)."""
    model_out_path = None
    if sentiment == "positive":
        model_out_path = "models/model_pos"
    elif sentiment == "negative":
        model_out_path = "models/model_neg"
    return model_out_path




## === cell 14
def get_training_data(sentiment):
    """Returns training data in spaCy NER format: (text, {"entities": [(start,end,label)]})."""
    train_data = []
    for _, row in df_train.iterrows():
        if row.sentiment != sentiment:
            continue
        text = str(row.text)
        selected_text = str(row.selected_text)

        start = text.find(selected_text)
        if start == -1:
            continue
        end = start + len(selected_text)
        if end <= start:
            continue

        train_data.append((text, {"entities": [(start, end, "selected_text")]}))
    return train_data




## === cell 15
def train(train_data, output_dir, n_iter=20, model=None):
    """Train spaCy NER and save to ./tse-spacy-model/{output_dir}."""
    if model is not None:
        nlp = spacy.load(f"./tse-spacy-model/{output_dir}")
        print(f"Loaded model '{model}'")
    else:
        nlp = spacy.blank("en")
        print("Created blank 'en' model")

    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        if model is None:
            nlp.initialize(get_examples=lambda: [])
        else:
            nlp.resume_training()

        for _ in tqdm(range(n_iter)):
            random.shuffle(train_data)
            batches = minibatch(train_data, size=compounding(4.0, 500.0, 1.001))
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,
                    annotations,
                    drop=0.5,
                    losses=losses,
                )
            print("Losses", losses)

    save_model(output_dir, nlp, "st_ner")




## === cell 16
sentiment = "positive"
train_data = get_training_data(sentiment)
model_path = get_model_out_path(sentiment)
train(train_data, model_path, n_iter=3, model=None)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1106030329.py in <cell line: 0>()
      2 train_data = get_training_data(sentiment)
      3 model_path = get_model_out_path(sentiment)
----> 4 train(train_data, model_path, n_iter=3, model=None)
      5 

/tmp/ipykernel_11/2661769127.py in train(train_data, output_dir, n_iter, model)
     22     with nlp.disable_pipes(*other_pipes):
     23         if model is None:
---> 24             nlp.initialize(get_examples=lambda: [])
     25         else:
     26             nlp.resume_training()

/usr/local/lib/python3.11/dist-packages/spacy/language.py in initialize(self, get_examples, sgd)
   1351                     proc.initialize, p_settings, section="components", name=name
   1352                 )
-> 1353                 proc.initialize(get_examples, nlp=self, **p_settings)
   1354         pretrain_cfg = config.get("pretraining")
   1355         if pretrain_cfg:

/usr/local/lib/python3.11/dist-packages/spacy/pipeline/transition_parser.pyx in spacy.pipeline.transition_parser.Parser.initialize()

/usr/local/lib/python3.11/dist-packages/spacy/training/example.pyx in spacy.training.example.validate_get_examples()

TypeError: [E930] Received invalid get_examples callback in `Parser.initialize`. Expected function that returns an iterable of Example objects but got: []

## === cell 17
sentiment = "negative"
train_data = get_training_data(sentiment)
model_path = get_model_out_path(sentiment)
train(train_data, model_path, n_iter=3, model=None)




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/728385249.py in <cell line: 0>()
      2 train_data = get_training_data(sentiment)
      3 model_path = get_model_out_path(sentiment)
----> 4 train(train_data, model_path, n_iter=3, model=None)
      5 
      6 

/tmp/ipykernel_11/2661769127.py in train(train_data, output_dir, n_iter, model)
     22     with nlp.disable_pipes(*other_pipes):
     23         if model is None:
---> 24             nlp.initialize(get_examples=lambda: [])
     25         else:
     26             nlp.resume_training()

/usr/local/lib/python3.11/dist-packages/spacy/language.py in initialize(self, get_examples, sgd)
   1351                     proc.initialize, p_settings, section="components", name=name
   1352                 )
-> 1353                 proc.initialize(get_examples, nlp=self, **p_settings)
   1354         pretrain_cfg = config.get("pretraining")
   1355         if pretrain_cfg:

/usr/local/lib/python3.11/dist-packages/spacy/pipeline/transition_parser.pyx in spacy.pipeline.transition_parser.Parser.initialize()

/usr/local/lib/python3.11/dist-packages/spacy/training/example.pyx in spacy.training.example.validate_get_examples()

TypeError: [E930] Received invalid get_examples callback in `Parser.initialize`. Expected function that returns an iterable of Example objects but got: []

## === cell 18
def predict_entities(text, model):
    text = str(text)
    doc = model(text)
    ent_array = []
    for ent in doc.ents:
        start = text.find(ent.text)
        end = start + len(ent.text)
        new_int = [start, end, ent.label_]
        if start != -1 and end > start and new_int not in ent_array:
            ent_array.append(new_int)
    selected_text = (
        text[ent_array[0][0] : ent_array[0][1]] if len(ent_array) > 0 else text
    )
    return selected_text




## === cell 19
selected_texts = []
MODELS_BASE_PATH = "./tse-spacy-model/models/"

print("Loading Models from", MODELS_BASE_PATH)
model_pos = spacy.load(MODELS_BASE_PATH + "model_pos")
model_neg = spacy.load(MODELS_BASE_PATH + "model_neg")

for _, row in df_test.iterrows():
    text = str(row.text)
    if row.sentiment == "neutral" or len(text.split()) <= 2:
        selected_texts.append(text)
    elif row.sentiment == "positive":
        selected_texts.append(predict_entities(text, model_pos))
    else:
        selected_texts.append(predict_entities(text, model_neg))

df_test["selected_text"] = selected_texts



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/2154959041.py in <cell line: 0>()
      3 
      4 print("Loading Models from", MODELS_BASE_PATH)
----> 5 model_pos = spacy.load(MODELS_BASE_PATH + "model_pos")
      6 model_neg = spacy.load(MODELS_BASE_PATH + "model_neg")
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

OSError: [E050] Can't find model './tse-spacy-model/models/model_pos'. It doesn't seem to be a Python package or a valid path to a data directory.

## === cell 20
sub = df_submission[["textID"]].merge(
    df_test[["textID", "selected_text"]], on="textID", how="left"
)
sub["selected_text"] = sub["selected_text"].fillna("")
sub.to_csv("submission.csv", index=False)
print(sub.head(10))
print("Wrote submission.csv with shape:", sub.shape)

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/150303380.py in <cell line: 0>()
      1 # Ensure submission format and row alignment by textID
      2 sub = df_submission[["textID"]].merge(
----> 3     df_test[["textID", "selected_text"]], on="textID", how="left"
      4 )
      5 sub["selected_text"] = sub["selected_text"].fillna("")

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['selected_text'] not in index"
