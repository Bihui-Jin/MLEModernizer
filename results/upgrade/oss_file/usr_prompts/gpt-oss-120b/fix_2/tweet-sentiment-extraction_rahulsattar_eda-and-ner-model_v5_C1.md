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

0.59156

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np, pandas as pd, os, re, math, random, tqdm, spacy
from collections import Counter
from tqdm import tqdm
from spacy.training.example import Example
from spacy.util import compounding, minibatch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


## === cell 1
train = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
print(f"Training Examples : {len(train)}")
print(f"Testing Examples : {len(test)}")


## === cell 2
WORD = re.compile(r"\w+")


def text_to_vector(text):
    return Counter(WORD.findall(text))


def get_cosine(text1, text2):
    vec1, vec2 = text_to_vector(text1), text_to_vector(text2)
    intersection = set(vec1) & set(vec2)
    numerator = sum(vec1[x] * vec2[x] for x in intersection)
    sum1 = sum(v**2 for v in vec1.values())
    sum2 = sum(v**2 for v in vec2.values())
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    return 0.0 if not denominator else float(numerator) / denominator




## === cell 3
train["COSINE_Score"] = train.apply(
    lambda r: get_cosine(str(r["text"]), str(r["selected_text"])), axis=1
)




## === cell 4
def save_model(output_dir, nlp, new_model_name):
    output_dir = f"../working/{output_dir}"
    os.makedirs(output_dir, exist_ok=True)
    nlp.meta["name"] = new_model_name
    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)




## === cell 5
def train_model(train_data, output_dir, n_iter=20, model=None):
    """Train a spaCy NER model on the provided data."""
    if model is not None:
        nlp = spacy.load(output_dir)
        print(f"Loaded model '{model}'")
    else:
        nlp = spacy.blank("en")
        print("Created blank 'en' model")
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")
    for _, ann in train_data:
        for start, end, label in ann.get("entities"):
            ner.add_label(label)
    other_pipes = [p for p in nlp.pipe_names if p != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.initialize()
        for itn in tqdm(range(n_iter)):
            random.shuffle(train_data)
            batches = minibatch(train_data, size=compounding(4.0, 500.0, 1.001))
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                docs = [nlp.make_doc(t) for t in texts]
                examples = [
                    Example.from_dict(doc, ann) for doc, ann in zip(docs, annotations)
                ]
                nlp.update(
                    examples,
                    drop=0.5,
                    losses=losses,
                    optimizer=optimizer,
                )
            print(f"Iteration {itn+1} Losses:", losses)
    save_model(output_dir, nlp, "st_ner")




## === cell 6
def get_training_data(df):
    data = []
    for _, row in df.iterrows():
        sentiment = row.sentiment
        selected = str(row.selected_text)
        text = str(row.text)
        start = text.find(selected)
        end = start + len(selected)
        data.append((text, {"entities": [[start, end, sentiment]]}))
    return data




## === cell 7
model_path = "models/model"
train_data = get_training_data(train)
train_model(train_data, model_path, n_iter=3, model=None)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/365111445.py in <cell line: 0>()
      1 model_path = "models/model"
      2 train_data = get_training_data(train)
----> 3 train_model(train_data, model_path, n_iter=3, model=None)
      4 
      5 

/tmp/ipykernel_11/1844644086.py in train_model(train_data, output_dir, n_iter, model)
     30                     Example.from_dict(doc, ann) for doc, ann in zip(docs, annotations)
     31                 ]
---> 32                 nlp.update(
     33                     examples,
     34                     drop=0.5,

TypeError: Language.update() got an unexpected keyword argument 'optimizer'

## === cell 8
def predict_entities(text, model):
    doc = model(text)
    ents = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
    if not ents:
        return text
    start, end, _ = ents[0]
    return text[start:end]




## === cell 9
selected_texts = []
model_dir = "/kaggle/working/models/model"
print("Loading model from", model_dir)
model = spacy.load(model_dir)
for _, row in test.iterrows():
    selected_texts.append(predict_entities(row.text, model))
test["selected_text"] = selected_texts


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/2113293731.py in <cell line: 0>()
      2 model_dir = "/kaggle/working/models/model"
      3 print("Loading model from", model_dir)
----> 4 model = spacy.load(model_dir)
      5 for _, row in test.iterrows():
      6     selected_texts.append(predict_entities(row.text, model))

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

OSError: [E050] Can't find model '/kaggle/working/models/model'. It doesn't seem to be a Python package or a valid path to a data directory.

## === cell 10
submission = pd.read_csv(
    "/kaggle/input/tweet-sentiment-extraction/sample_submission.csv"
)
submission["selected_text"] = test["selected_text"]
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")
display(submission.head())

## --- ERROR in cell 10, traceback:
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
/tmp/ipykernel_11/2893010774.py in <cell line: 0>()
      2     "/kaggle/input/tweet-sentiment-extraction/sample_submission.csv"
      3 )
----> 4 submission["selected_text"] = test["selected_text"]
      5 submission.to_csv("submission.csv", index=False)
      6 print("Saved submission.csv")

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
