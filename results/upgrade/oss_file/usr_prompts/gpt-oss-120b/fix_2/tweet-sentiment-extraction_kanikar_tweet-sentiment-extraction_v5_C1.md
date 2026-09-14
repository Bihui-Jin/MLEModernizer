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

3.10

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

0.5854182243347168

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
import random
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import string
from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.corpus import stopwords
import spacy
from spacy.util import compounding, minibatch
from tqdm import tqdm
import os
import warnings

warnings.filterwarnings("ignore")


## === cell 1
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


## === cell 2
df_train = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
df_test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")


## === cell 3
df_train.dropna(inplace=True)




## === cell 4
def clean_text(text):
    text = str(text).lower()
    text = re.sub("\[.*?\]", "", text)
    text = re.sub("https?://\S+|www\.\S+", "", text)
    text = re.sub("<.*?>+", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub("\n", "", text)
    text = re.sub("\w*\d\w*", "", text)
    return text


df_train["text"] = df_train["text"].apply(clean_text)
df_train["selected_text"] = df_train["selected_text"].apply(clean_text)




## === cell 5
def get_training_data():
    train_data = []
    for _, row in df_train.iterrows():
        text = row.text
        selected_text = row.selected_text
        start = text.find(selected_text)
        if start == -1:
            start = 0
        end = start + len(selected_text)
        train_data.append((text, {"entities": [[start, end, "selected_text"]]}))
    return train_data




## === cell 6
def get_model_out_path(sentiment):
    if sentiment == "positive":
        return "models/model_pos"
    elif sentiment == "negative":
        return "models/model_neg"
    else:
        return None




## === cell 7
def trim_entity_spans(data: list) -> list:
    invalid_span_tokens = re.compile(r"\s")
    cleaned_data = []
    for text, annotations in data:
        entities = annotations["entities"]
        valid_entities = []
        for start, end, label in entities:
            valid_start = start
            valid_end = end
            while valid_start < len(text) and invalid_span_tokens.match(
                text[valid_start]
            ):
                valid_start += 1
            while valid_end > 1 and invalid_span_tokens.match(text[valid_end - 1]):
                valid_end -= 1
            valid_entities.append([valid_start, valid_end, label])
        cleaned_data.append([text, {"entities": valid_entities}])
    return cleaned_data




## === cell 8
def train(train_data, output_dir, n_iter=20, model=None):
    train_data = trim_entity_spans(train_data)
    if model is not None:
        nlp = spacy.load(model)
        print(f"Loaded model '{model}'")
    else:
        nlp = spacy.blank("en")
        print("Created blank 'en' model")
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")
    for _, annotations in train_data:
        for start, end, label in annotations.get("entities"):
            ner.add_label(label)
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for itn in tqdm(range(n_iter), desc="Training"):
            random.shuffle(train_data)
            losses = {}
            for text, annotations in train_data:
                try:
                    nlp.update(
                        [text], [annotations], drop=0.2, sgd=optimizer, losses=losses
                    )
                except Exception:
                    continue
            print(f"Iteration {itn+1} losses:", losses)
    os.makedirs(output_dir, exist_ok=True)
    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)




## === cell 9
def predict_entities(text, model):
    doc = model(text)
    ent_array = []
    for ent in doc.ents:
        start = text.find(ent.text)
        end = start + len(ent.text)
        if start != -1:
            entry = [start, end, ent.label_]
            if entry not in ent_array:
                ent_array.append(entry)
    if ent_array:
        start, end, _ = ent_array[0]
        return text[start:end]
    else:
        return text




## === cell 10
sentiment = "positive"
train_data_pos = get_training_data()
model_path_pos = get_model_out_path(sentiment)
train(train_data_pos, model_path_pos, n_iter=2, model=None)
sentiment = "negative"
train_data_neg = get_training_data()
model_path_neg = get_model_out_path(sentiment)
train(train_data_neg, model_path_neg, n_iter=2, model=None)


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/3077166771.py in <cell line: 0>()
      3 train_data_pos = get_training_data()
      4 model_path_pos = get_model_out_path(sentiment)
----> 5 train(train_data_pos, model_path_pos, n_iter=2, model=None)
      6 # Train negative model
      7 sentiment = "negative"

/tmp/ipykernel_11/237097345.py in train(train_data, output_dir, n_iter, model)
      1 def train(train_data, output_dir, n_iter=20, model=None):
----> 2     train_data = trim_entity_spans(train_data)
      3     if model is not None:
      4         nlp = spacy.load(model)
      5         print(f"Loaded model '{model}'")

/tmp/ipykernel_11/3696043351.py in trim_entity_spans(data)
     12             ):
     13                 valid_start += 1
---> 14             while valid_end > 1 and invalid_span_tokens.match(text[valid_end - 1]):
     15                 valid_end -= 1
     16             valid_entities.append([valid_start, valid_end, label])

IndexError: string index out of range

## === cell 11
model_pos = spacy.load("models/model_pos")
model_neg = spacy.load("models/model_neg")


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/2692993324.py in <cell line: 0>()
      1 # Load models
----> 2 model_pos = spacy.load("models/model_pos")
      3 model_neg = spacy.load("models/model_neg")

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

OSError: [E050] Can't find model 'models/model_pos'. It doesn't seem to be a Python package or a valid path to a data directory.

## === cell 12
selected_texts = []
for _, row in df_test.iterrows():
    text = row.text
    if row.sentiment == "neutral" or len(text.split()) <= 2:
        selected_texts.append(text)
    elif row.sentiment == "positive":
        selected_texts.append(predict_entities(text, model_pos))
    else:
        selected_texts.append(predict_entities(text, model_neg))
df_test["selected_text"] = selected_texts


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2189981429.py in <cell line: 0>()
      5         selected_texts.append(text)
      6     elif row.sentiment == "positive":
----> 7         selected_texts.append(predict_entities(text, model_pos))
      8     else:
      9         selected_texts.append(predict_entities(text, model_neg))

NameError: name 'model_pos' is not defined

## === cell 13
submission = pd.DataFrame(
    {"textID": df_test["textID"], "selected_text": df_test["selected_text"]}
)
submission.to_csv("submission.csv", index=False)
print("Submission file written to submission.csv")
display(submission.head(10))

## --- ERROR in cell 13, traceback:
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
/tmp/ipykernel_11/3538393662.py in <cell line: 0>()
      1 submission = pd.DataFrame(
----> 2     {"textID": df_test["textID"], "selected_text": df_test["selected_text"]}
      3 )
      4 submission.to_csv("submission.csv", index=False)
      5 print("Submission file written to submission.csv")

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
