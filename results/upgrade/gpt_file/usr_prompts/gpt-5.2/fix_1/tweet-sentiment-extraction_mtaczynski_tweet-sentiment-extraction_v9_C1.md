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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
spacy==3.8.7
spacy-legacy==3.0.12
spacy-loggers==1.0.5
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1
transformers==4.53.3

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

0.62529

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import random
import torch
import os
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from transformers import BertTokenizer
from tqdm.notebook import tqdm_notebook
from sklearn.preprocessing import OneHotEncoder
import re
tqdm_notebook.pandas()


## === cell 1
if torch.cuda.is_available():    

    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


## === cell 2
try:
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')
except:
    df_train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
    df_test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')


## === cell 3
df_train["text_original"] = df_train["text"]


## === cell 4
df_train.fillna("", inplace=True)
df_test.fillna("", inplace=True)


## === cell 6
lb = LabelEncoder()
df_train["target"] = lb.fit_transform(df_train["sentiment"])


## === cell 7
tf = TfidfVectorizer(ngram_range=(1,1))
X_train, X_val, y_train, y_val =  train_test_split(tf.fit_transform(df_train["text"]),df_train["target"])
df_train["is_training"] = [1 if x in y_train.index else 0 for x in df_train.index]


## === cell 8
parameters = {}

clf = OneVsRestClassifier(LogisticRegression(solver="lbfgs"))
clf.fit(X_train, y_train)


## === cell 9
y_val_predict_sentiment = clf.predict(X_val)


## === cell 10
f1_score(y_val, y_val_predict_sentiment, average='weighted')


## === cell 11
import spacy
from spacy.util import compounding, minibatch

train_data = []
for idx, row in df_train[(df_train["sentiment"]!="neutral")].iterrows():
    text = row["text"]
    selected_text = row["selected_text"]
    if selected_text in text:
        entities = []
        try:
            for match in re.finditer(re.escape(selected_text), text):
                start_char = match.start()
                end_char = match.end()
                entity_label = row["sentiment"]
                entities.append((start_char,end_char, entity_label))
        except Exception as e:
            print(text)
            print(selected_text)
            raise e
        train_data.append((text,{"entities":entities}))


## === cell 12
def spacy_train_custom(train_data, epochs=10):
    sample_print = 2
    
    nlp = spacy.blank("en")  # create blank Language class

    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    else:
        ner = nlp.get_pipe("ner")

    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        nlp.begin_training()
        n_iterations = epochs
        for itn in range(n_iterations):
            random.shuffle(train_data)
            losses = {}
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001)) # 1.001
            for batch_no, batch in enumerate(batches):
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.3,
                    losses=losses,
                )
            if sample_print > 0:
                print("Batch {} out of {}. Losses: {}".format(itn + 1, n_iterations, losses))

    for text, _ in train_data[:sample_print]:
        doc = nlp(text)
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
        print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])
        
    return nlp


## === cell 13
nlp = spacy_train_custom(train_data[0:1000], epochs=20)


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/344130396.py in <cell line: 0>()
----> 1 nlp = spacy_train_custom(train_data[0:1000], epochs=20)

/tmp/ipykernel_11/3726134282.py in spacy_train_custom(train_data, epochs)
      8     if "ner" not in nlp.pipe_names:
      9         ner = nlp.create_pipe("ner")
---> 10         nlp.add_pipe(ner, last=True)
     11     # otherwise, get it so we can add labels
     12     else:

/usr/local/lib/python3.11/dist-packages/spacy/language.py in add_pipe(self, factory_name, name, before, after, first, last, source, config, raw_config, validate)
    809             bad_val = repr(factory_name)
    810             err = Errors.E966.format(component=bad_val, name=name)
--> 811             raise ValueError(err)
    812         name = name if name is not None else factory_name
    813         if name in self.component_names:

ValueError: [E966] `nlp.add_pipe` now takes the string name of the registered component factory, not a callable component. Expected string, but got <spacy.pipeline.ner.EntityRecognizer object at 0x7f1bd3156650> (name: 'None').

- If you created your component with `nlp.create_pipe('name')`: remove nlp.create_pipe and call `nlp.add_pipe('name')` instead.

- If you passed in a component like `TextCategorizer()`: call `nlp.add_pipe` with the string name instead, e.g. `nlp.add_pipe('textcat')`.

- If you're using a custom component: Add the decorator `@Language.component` (for function components) or `@Language.factory` (for class components / factories) to your custom component and assign it a name, e.g. `@Language.component('your_name')`. You can then run `nlp.add_pipe('your_name')` to add it to the pipeline.

## === cell 14
from spacy import displacy
sample = df_train.sample().iloc[0]
doc = nlp(sample.text)
displacy.render(doc, style="ent")
print(sample.sentiment)
print(sample.selected_text)


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2399904044.py in <cell line: 0>()
      1 from spacy import displacy
      2 sample = df_train.sample().iloc[0]
----> 3 doc = nlp(sample.text)
      4 displacy.render(doc, style="ent")
      5 print(sample.sentiment)

NameError: name 'nlp' is not defined

## === cell 15
df_test["selected_text"] = df_test["text"].progress_apply(lambda x: " ".join([l.text for l in nlp(x).ents]))
df_test["selected_text"] = [row["text"] if row["sentiment"] == "neutral" or row["selected_text"] == "" else row["selected_text"] for idx, row in df_test.iterrows()]


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2215754526.py in <cell line: 0>()
----> 1 df_test["selected_text"] = df_test["text"].progress_apply(lambda x: " ".join([l.text for l in nlp(x).ents]))
      2 df_test["selected_text"] = [row["text"] if row["sentiment"] == "neutral" or row["selected_text"] == "" else row["selected_text"] for idx, row in df_test.iterrows()]

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in inner(df, func, *args, **kwargs)
    915                 # on the df using our wrapper (which provides bar updating)
    916                 try:
--> 917                     return getattr(df, df_function)(wrapper, **kwargs)
    918                 finally:
    919                     t.close()

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in apply(self, func, convert_dtype, args, by_row, **kwargs)
   4922             args=args,
   4923             kwargs=kwargs,
-> 4924         ).apply()
   4925 
   4926     def _reindex_indexer(

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply(self)
   1425 
   1426         # self.func is Callable
-> 1427         return self.apply_standard()
   1428 
   1429     def agg(self):

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_standard(self)
   1505         #  Categorical (GH51645).
   1506         action = "ignore" if isinstance(obj.dtype, CategoricalDtype) else None
-> 1507         mapped = obj._map_values(
   1508             mapper=curried, na_action=action, convert=self.convert_dtype
   1509         )

/usr/local/lib/python3.11/dist-packages/pandas/core/base.py in _map_values(self, mapper, na_action, convert)
    919             return arr.map(mapper, na_action=na_action)
    920 
--> 921         return algorithms.map_array(arr, mapper, na_action=na_action, convert=convert)
    922 
    923     @final

/usr/local/lib/python3.11/dist-packages/pandas/core/algorithms.py in map_array(arr, mapper, na_action, convert)
   1741     values = arr.astype(object, copy=False)
   1742     if na_action is None:
-> 1743         return lib.map_infer(values, mapper, convert=convert)
   1744     else:
   1745         return lib.map_infer_mask(

lib.pyx in pandas._libs.lib.map_infer()

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in wrapper(*args, **kwargs)
    910                     # take a fast or slow code path; so stop when t.total==t.n
    911                     t.update(n=1 if not t.total or t.n < t.total else 0)
--> 912                     return func(*args, **kwargs)
    913 
    914                 # Apply the provided function (in **kwargs)

/tmp/ipykernel_11/2215754526.py in <lambda>(x)
----> 1 df_test["selected_text"] = df_test["text"].progress_apply(lambda x: " ".join([l.text for l in nlp(x).ents]))
      2 df_test["selected_text"] = [row["text"] if row["sentiment"] == "neutral" or row["selected_text"] == "" else row["selected_text"] for idx, row in df_test.iterrows()]

NameError: name 'nlp' is not defined

## === cell 16
df_test[["textID","selected_text"]].to_csv("submission.csv",index=False)


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1443694171.py in <cell line: 0>()
----> 1 df_test[["textID","selected_text"]].to_csv("submission.csv",index=False)

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
