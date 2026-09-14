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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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

0.62253

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
TRAIN_PATH = "/kaggle/input/tweet-sentiment-extraction/train.csv"
TEST_PATH = "/kaggle/input/tweet-sentiment-extraction/test.csv"
SAMPLE_SUB_PATH = "/kaggle/input/tweet-sentiment-extraction/sample_submission.csv"

df_test = pd.read_csv(TEST_PATH)
df_test.head()



## === cell 2
df_train = pd.read_csv(TRAIN_PATH)
df_train.head()



## === cell 3
print(len(df_train))
df_train.dropna(axis=0, how="any", inplace=True)
print(len(df_train))



## === cell 4
df_train["text_tokes"] = df_train.text.str.split()
df_train["select_tokes"] = df_train.selected_text.str.split()
df_train["text_tokes_cnt"] = df_train.text_tokes.str.len()
df_train["select_tokes_cnt"] = df_train.select_tokes.str.len()
df_train.head(5)



## === cell 5
df_train = df_train[~(df_train.text_tokes_cnt <= 2)]
df_train = df_train[(df_train.sentiment != "neutral")]
print(len(df_train))
df_train.sentiment.value_counts()



## === cell 6
import spacy
from tqdm import tqdm
import random
from spacy.util import minibatch, compounding

import warnings

warnings.filterwarnings("ignore")




## === cell 7
def save_model(output_dir, nlp, new_model_name):
    """Save spaCy model to a directory path."""
    if output_dir is None:
        return
    os.makedirs(output_dir, exist_ok=True)
    nlp.meta["name"] = new_model_name
    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)




## === cell 8
def get_model_out_path(sentiment):
    """Returns Model output path."""
    if sentiment == "positive":
        return "/kaggle/working/models/model_pos"
    if sentiment == "negative":
        return "/kaggle/working/models/model_neg"
    return None




## === cell 9
def get_training_data(sentiment, df_input):
    """
    Returns Training data in the format needed to train spacy NER.
    Use start/end offsets of selected_text inside text.
    """
    SENTIMENT = ["negative", "positive"]
    if sentiment not in SENTIMENT:
        raise ValueError(f"{sentiment} not in {SENTIMENT})")
    train_data = []
    for _, row in df_input.iterrows():
        if row.sentiment == sentiment:
            selected_text = row.selected_text
            text = row.text
            start = text.find(selected_text)
            if start == -1:
                continue
            end = start + len(selected_text)
            train_data.append((text, {"entities": [(start, end, "selected_text")]}))
    return train_data




## === cell 10
def train(train_data, output_dir, n_iter=20, model=None):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None and output_dir is not None and os.path.exists(output_dir):
        nlp = spacy.load(output_dir)
        print("Loaded model '%s'" % model)
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
            optimizer = nlp.begin_training()
        else:
            optimizer = nlp.resume_training()

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
                    sgd=optimizer,
                    losses=losses,
                )
            print("Losses", losses)

    save_model(output_dir, nlp, "st_ner")




## === cell 11
def run_train(n_iter=3):
    """Convenience wrapper to train models for both sentiments."""
    for sentiment in ["positive", "negative"]:
        model_path = get_model_out_path(sentiment)
        train_data = get_training_data(sentiment, df_train)
        train(train_data, model_path, n_iter=n_iter)




## === cell 12
run_train(n_iter=1)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3634862468.py in <cell line: 0>()
      1 # Kaggle cell magic removed for .py compatibility; keep same effective behavior.
----> 2 run_train(n_iter=1)
      3 
      4 

/tmp/ipykernel_11/3513797339.py in run_train(n_iter)
      4         model_path = get_model_out_path(sentiment)
      5         train_data = get_training_data(sentiment, df_train)
----> 6         train(train_data, model_path, n_iter=n_iter)
      7 
      8 

/tmp/ipykernel_11/3669625991.py in train(train_data, output_dir, n_iter, model)
     31             for batch in batches:
     32                 texts, annotations = zip(*batch)
---> 33                 nlp.update(
     34                     texts,
     35                     annotations,

/usr/local/lib/python3.11/dist-packages/spacy/language.py in update(self, examples, _, drop, sgd, losses, component_cfg, exclude, annotates)
   1173         """
   1174         if _ is not None:
-> 1175             raise ValueError(Errors.E989)
   1176         if losses is None:
   1177             losses = {}

ValueError: [E989] `nlp.update()` was called with two positional arguments. This may be due to a backwards-incompatible change to the format of the training data in spaCy 3.0 onwards. The 'update' function should now be called with a batch of Example objects, instead of `(text, annotation)` tuples. 

## === cell 13
def predict_entities(text, model):
    doc = model(text)
    if not doc.ents:
        return text
    best_ent = max(doc.ents, key=lambda e: (len(e.text), -e.start_char))
    return text[best_ent.start_char : best_ent.end_char]




## === cell 14
selected_texts = []
print("Loading Models from /kaggle/working/models/")
model_pos_path = get_model_out_path("positive")
model_neg_path = get_model_out_path("negative")

model_pos = spacy.load(model_pos_path)
model_neg = spacy.load(model_neg_path)

for _, row in df_test.iterrows():
    text = row.text
    if row.sentiment == "neutral" or len(str(text).split()) <= 2:
        selected_texts.append(text)
    elif row.sentiment == "positive":
        selected_texts.append(predict_entities(text, model_pos))
    else:
        selected_texts.append(predict_entities(text, model_neg))

df_test["selected_text"] = selected_texts



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/4111857364.py in <cell line: 0>()
      4 model_neg_path = get_model_out_path("negative")
      5 
----> 6 model_pos = spacy.load(model_pos_path)
      7 model_neg = spacy.load(model_neg_path)
      8 

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

## === cell 15
print(len(df_test))
df_test.sentiment.value_counts()



## === cell 16
pd.options.display.max_colwidth = 1000
df_test[df_test.sentiment.isin(["positive"])].sample(10, random_state=0)



## === cell 17
print(len(df_test))
df_test.columns



## === cell 18
df_submission = df_test[["textID", "selected_text"]].copy()
df_submission["selected_text"] = df_submission["selected_text"].fillna("").astype(str)
print(len(df_submission))
df_submission.head(10)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1122341292.py in <cell line: 0>()
      1 # Ensure required columns exist and are aligned with sample submission ids.
----> 2 df_submission = df_test[["textID", "selected_text"]].copy()
      3 # Minimal sanitation: ensure strings (keeps evaluation semantics; avoids NaNs breaking CSV quoting)
      4 df_submission["selected_text"] = df_submission["selected_text"].fillna("").astype(str)
      5 print(len(df_submission))

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

## === cell 19
SUB_PATH = "/kaggle/working/submission.csv"
df_submission.to_csv(SUB_PATH, index=False)
print("Wrote:", SUB_PATH)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3473357775.py in <cell line: 0>()
      1 # Write submission to the standard working directory with required filename/extension.
      2 SUB_PATH = "/kaggle/working/submission.csv"
----> 3 df_submission.to_csv(SUB_PATH, index=False)
      4 print("Wrote:", SUB_PATH)
      5 

NameError: name 'df_submission' is not defined

## === cell 20
print(os.listdir("/kaggle/working")[:50])
