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

0.62535

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
import re
import os
import random
from tqdm import tqdm
import spacy
from spacy.util import minibatch, compounding

for dirname, _, filenames in os.walk("/kaggle/working"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_data = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
train_data.dropna(inplace=True)
test_data = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")




## === cell 1
def get_training_data(sentiment):
    """Create spaCy training format for a given sentiment."""
    train_df = []
    for _, row in train_data.iterrows():
        if row.sentiment == sentiment:
            selected_text = row.selected_text
            text = row.text
            start = text.find(selected_text)
            end = start + len(selected_text)
            train_df.append((text, {"entities": [[start, end, "selected_text"]]}))
    return train_df


def get_model_out_path(sentiment):
    """Return the directory where a model for *sentiment* will be saved."""
    if sentiment == "positive":
        return "/kaggle/working/model_pos"
    if sentiment == "negative":
        return "/kaggle/working/model_neg"
    if sentiment == "neutral":
        return "/kaggle/working/model_neu"
    return None


def save_model(output_dir, nlp, new_model_name):
    """Save the trained spaCy model to *output_dir*."""
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        nlp.meta["name"] = new_model_name
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)


def train(train_data, output_dir, n_iter=5, model=None):
    """Train a blank spaCy NER model on *train_data*."""
    if model is not None:
        nlp = spacy.load(output_dir)
        print("Loaded existing model")
    else:
        nlp = spacy.blank("en")
        print("Created blank 'en' model")

    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        if model is None:
            nlp.begin_training()
        else:
            nlp.resume_training()
        for itn in tqdm(range(n_iter), desc="Training iterations"):
            random.shuffle(train_data)
            batches = minibatch(train_data, size=compounding(4.0, 500.0, 1.001))
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, drop=0.5, losses=losses)
            print("Iteration", itn, "Losses", losses)
    save_model(output_dir, nlp, "st_ner")
    return nlp


def predict_entities(text, model):
    """Return the extracted selected text using the trained NER model."""
    doc = model(text)
    ent_array = []
    for ent in doc.ents:
        start = text.find(ent.text)
        end = start + len(ent.text)
        if [start, end, ent.label_] not in ent_array:
            ent_array.append([start, end, ent.label_])
    if ent_array:
        return text[ent_array[0][0] : ent_array[0][1]]
    return text




## === cell 2
sentiments = ["positive", "negative", "neutral"]
trained_models = {}
for sentiment in sentiments:
    train_df = get_training_data(sentiment)
    model_path = get_model_out_path(sentiment)
    trained_models[sentiment] = train(train_df, model_path, n_iter=5, model=None)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1163136079.py in <cell line: 0>()
      5     train_df = get_training_data(sentiment)
      6     model_path = get_model_out_path(sentiment)
----> 7     trained_models[sentiment] = train(train_df, model_path, n_iter=5, model=None)
      8 

/tmp/ipykernel_11/1023772782.py in train(train_data, output_dir, n_iter, model)
     63             for batch in batches:
     64                 texts, annotations = zip(*batch)
---> 65                 nlp.update(texts, annotations, drop=0.5, losses=losses)
     66             print("Iteration", itn, "Losses", losses)
     67     save_model(output_dir, nlp, "st_ner")

/usr/local/lib/python3.11/dist-packages/spacy/language.py in update(self, examples, _, drop, sgd, losses, component_cfg, exclude, annotates)
   1173         """
   1174         if _ is not None:
-> 1175             raise ValueError(Errors.E989)
   1176         if losses is None:
   1177             losses = {}

ValueError: [E989] `nlp.update()` was called with two positional arguments. This may be due to a backwards-incompatible change to the format of the training data in spaCy 3.0 onwards. The 'update' function should now be called with a batch of Example objects, instead of `(text, annotation)` tuples. 

## === cell 3
model_pos = spacy.load(get_model_out_path("positive"))
model_neg = spacy.load(get_model_out_path("negative"))
model_neu = spacy.load(get_model_out_path("neutral"))


def predict_on_test(row):
    """Select the appropriate model based on sentiment and predict."""
    sentiment = row["sentiment"]
    text = row["text"]
    if sentiment == "positive":
        return predict_entities(text, model_pos)
    if sentiment == "negative":
        return predict_entities(text, model_neg)
    return predict_entities(text, model_neu)


test_data["selected_text"] = test_data.apply(predict_on_test, axis=1)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/1108306358.py in <cell line: 0>()
      1 # Load the trained models (they were saved in the previous step)
----> 2 model_pos = spacy.load(get_model_out_path("positive"))
      3 model_neg = spacy.load(get_model_out_path("negative"))
      4 model_neu = spacy.load(get_model_out_path("neutral"))
      5 

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

OSError: [E050] Can't find model '/kaggle/working/model_pos'. It doesn't seem to be a Python package or a valid path to a data directory.

## === cell 4
submission = test_data[["textID", "selected_text"]].copy()
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1609168349.py in <cell line: 0>()
      1 # Create submission file in the required format
----> 2 submission = test_data[["textID", "selected_text"]].copy()
      3 submission.to_csv("submission.csv", index=False)
      4 print("Submission saved to submission.csv")

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
