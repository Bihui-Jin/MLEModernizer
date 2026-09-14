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

0.65819

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
from pathlib import Path

import numpy as np
import pandas as pd

import spacy
from spacy.util import minibatch, compounding
from tqdm import tqdm

random.seed(42)
np.random.seed(42)

TRAIN_PATH = "/kaggle/input/tweet-sentiment-extraction/train.csv"
TEST_PATH = "/kaggle/input/tweet-sentiment-extraction/test.csv"

train_data = pd.read_csv(TRAIN_PATH).dropna().reset_index(drop=True)
test_data = pd.read_csv(TEST_PATH).reset_index(drop=True)

TRAINED_MODELS_BASE_PATH = "/kaggle/working/"  # where models are saved




## === cell 1
def get_training_data(df: pd.DataFrame, sentiment: str):
    """
    Create spaCy NER training tuples: (text, {"entities": [(start,end,label)]})
    Skip rows where selected_text isn't found (prevents invalid offsets).
    """
    train_examples = []
    for _, row in df.iterrows():
        if row.sentiment != sentiment:
            continue
        text = str(row.text)
        selected_text = str(row.selected_text)

        start = text.find(selected_text)
        if start == -1:
            continue
        end = start + len(selected_text)
        train_examples.append((text, {"entities": [(start, end, "selected_text")]}))
    return train_examples


def get_model_out_path(sentiment: str):
    if sentiment == "positive":
        return os.path.join(TRAINED_MODELS_BASE_PATH, "model_pos")
    if sentiment == "negative":
        return os.path.join(TRAINED_MODELS_BASE_PATH, "model_neg")
    if sentiment == "neutral":
        return os.path.join(TRAINED_MODELS_BASE_PATH, "model_neu")
    raise ValueError(f"Unknown sentiment: {sentiment}")


def save_model(output_dir: str, nlp, new_model_name: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    nlp.meta["name"] = new_model_name
    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)


def train_ner(train_examples, output_dir, n_iter=20, model=None):
    """
    spaCy v3-compatible NER training loop, keeping the original approach:
    - blank English model
    - add NER pipe
    - train for n_iter with dropout
    """
    if model is not None:
        nlp = spacy.load(output_dir)
        print(f"Loaded model '{model}' from {output_dir}")
    else:
        nlp = spacy.blank("en")
        print("Created blank 'en' model")

    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    ner.add_label("selected_text")

    other_pipes = [p for p in nlp.pipe_names if p != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.initialize(get_examples=lambda: train_examples)

        for itn in tqdm(range(n_iter)):
            random.shuffle(train_examples)
            batches = minibatch(train_examples, size=compounding(4.0, 500.0, 1.001))
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                examples = []
                for t, ann in zip(texts, annotations):
                    doc = nlp.make_doc(t)
                    examples.append(spacy.training.Example.from_dict(doc, ann))

                nlp.update(
                    examples,
                    drop=0.5,
                    sgd=optimizer,
                    losses=losses,
                )

            print("Losses", losses)

    save_model(output_dir, nlp, "st_ner")
    return nlp




## === cell 2
sentiments = ["positive", "negative", "neutral"]
for sentiment in sentiments:
    train_examples = get_training_data(train_data, sentiment)
    model_path = get_model_out_path(sentiment)
    train_ner(train_examples, model_path, n_iter=2, model=None)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3036330259.py in <cell line: 0>()
      5     model_path = get_model_out_path(sentiment)
      6     # Keep original n_iter=2 to preserve intended training regime / runtime
----> 7     train_ner(train_examples, model_path, n_iter=2, model=None)
      8 
      9 

/tmp/ipykernel_11/2746959034.py in train_ner(train_examples, output_dir, n_iter, model)
     62     with nlp.disable_pipes(*other_pipes):
     63         # spaCy v3: initialize with examples' labels
---> 64         optimizer = nlp.initialize(get_examples=lambda: train_examples)
     65 
     66         for itn in tqdm(range(n_iter)):

/usr/local/lib/python3.11/dist-packages/spacy/language.py in initialize(self, get_examples, sgd)
   1351                     proc.initialize, p_settings, section="components", name=name
   1352                 )
-> 1353                 proc.initialize(get_examples, nlp=self, **p_settings)
   1354         pretrain_cfg = config.get("pretraining")
   1355         if pretrain_cfg:

/usr/local/lib/python3.11/dist-packages/spacy/pipeline/transition_parser.pyx in spacy.pipeline.transition_parser.Parser.initialize()

/usr/local/lib/python3.11/dist-packages/spacy/training/example.pyx in spacy.training.example.validate_get_examples()

/usr/local/lib/python3.11/dist-packages/spacy/training/example.pyx in spacy.training.example.validate_examples()

TypeError: [E978] The Parser.initialize method takes a list of Example objects, but got: {<class 'tuple'>}

## === cell 3
def predict_entities(text: str, model):
    """
    Predict NER spans and select the first unique entity span.
    If none predicted, fallback to full text (original behavior).
    """
    text = str(text)
    doc = model(text)

    ent_array = []
    for ent in doc.ents:
        start = text.find(ent.text)
        if start == -1:
            continue
        end = start + len(ent.text)
        item = [start, end, ent.label_]
        if item not in ent_array:
            ent_array.append(item)

    selected_text = (
        text[ent_array[0][0] : ent_array[0][1]] if len(ent_array) > 0 else text
    )
    return selected_text


def jaccard(str1, str2):
    a = set(str(str1).lower().split())
    b = set(str(str2).lower().split())
    c = a.intersection(b)
    denom = len(a) + len(b) - len(c)
    return float(len(c)) / denom if denom else 0.0


print("Loading Models from", TRAINED_MODELS_BASE_PATH)
model_pos = spacy.load(os.path.join(TRAINED_MODELS_BASE_PATH, "model_pos"))
model_neg = spacy.load(os.path.join(TRAINED_MODELS_BASE_PATH, "model_neg"))
model_neu = spacy.load(os.path.join(TRAINED_MODELS_BASE_PATH, "model_neu"))




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/491178914.py in <cell line: 0>()
     32 
     33 print("Loading Models from", TRAINED_MODELS_BASE_PATH)
---> 34 model_pos = spacy.load(os.path.join(TRAINED_MODELS_BASE_PATH, "model_pos"))
     35 model_neg = spacy.load(os.path.join(TRAINED_MODELS_BASE_PATH, "model_neg"))
     36 model_neu = spacy.load(os.path.join(TRAINED_MODELS_BASE_PATH, "model_neu"))

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
def predict_on_test_df(text, sentiment):
    if sentiment == "neutral":
        return predict_entities(text, model_neu)
    if sentiment == "positive":
        return predict_entities(text, model_pos)
    return predict_entities(text, model_neg)


test_data["selected_text"] = test_data.apply(
    lambda x: predict_on_test_df(x["text"], x["sentiment"]), axis=1
)

data_to_submit = test_data[["textID", "selected_text"]].copy()

out_path = "submission.csv"
data_to_submit.to_csv(out_path, index=False)
print("Wrote", out_path, "with shape", data_to_submit.shape)
print(data_to_submit.head())

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3025488116.py in <cell line: 0>()
      7 
      8 
----> 9 test_data["selected_text"] = test_data.apply(
     10     lambda x: predict_on_test_df(x["text"], x["sentiment"]), axis=1
     11 )

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in apply(self, func, axis, raw, result_type, args, by_row, engine, engine_kwargs, **kwargs)
  10372             kwargs=kwargs,
  10373         )
> 10374         return op.apply().__finalize__(self, method="apply")
  10375 
  10376     def map(

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply(self)
    914             return self.apply_raw(engine=self.engine, engine_kwargs=self.engine_kwargs)
    915 
--> 916         return self.apply_standard()
    917 
    918     def agg(self):

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_standard(self)
   1061     def apply_standard(self):
   1062         if self.engine == "python":
-> 1063             results, res_index = self.apply_series_generator()
   1064         else:
   1065             results, res_index = self.apply_series_numba()

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_series_generator(self)
   1079             for i, v in enumerate(series_gen):
   1080                 # ignore SettingWithCopy here in case the user mutates
-> 1081                 results[i] = self.func(v, *self.args, **self.kwargs)
   1082                 if isinstance(results[i], ABCSeries):
   1083                     # If we have a view on v, we need to make a copy because

/tmp/ipykernel_11/3025488116.py in <lambda>(x)
      8 
      9 test_data["selected_text"] = test_data.apply(
---> 10     lambda x: predict_on_test_df(x["text"], x["sentiment"]), axis=1
     11 )
     12 

/tmp/ipykernel_11/3025488116.py in predict_on_test_df(text, sentiment)
      3         return predict_entities(text, model_neu)
      4     if sentiment == "positive":
----> 5         return predict_entities(text, model_pos)
      6     return predict_entities(text, model_neg)
      7 

NameError: name 'model_pos' is not defined
