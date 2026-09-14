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

0.587620198726654

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
import os
import nltk
import re
from tqdm import tqdm
import spacy
from spacy.util import compounding
from spacy.util import minibatch

import warnings
warnings.filterwarnings("ignore")

## === cell 1
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

## === cell 2
train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
sample = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')

## === cell 3
train.dropna(inplace = True)

## === cell 4
train_data = train.copy()

## === cell 7
def get_training_data(sentiment):
    training_data = []
    for data in train_data.values:  
        if data[3] == sentiment:
            text = data[1]
            selected_text = data[2]
            start = text.find(selected_text)
            end = start + len(selected_text)
            training_data.append((text, {'entities' : [[start, end, 'selected_text']]}))
    return training_data

## === cell 8
def get_model_out_path(sentiment):
    model_out_path = None
    if sentiment == 'positive':
        model_out_path = 'models/model_pos'
    elif sentiment == 'negative':
        model_out_path = 'models/model_neg'
    else:
        model_out_path = 'models/model_neu'
    return model_out_path

## === cell 9
def save_model(output_dir, nlp, new_model_name):
    output_dir = f'/kaggle/input/tse-spacy-model/{output_dir}'
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        nlp.meta['name'] = new_model_name
        nlp.to_disk(output_dir)
        print("Saved model to ", output_dir)

## === cell 10
def train(training_data, output_dir, n_iter = 20, model = None):
    if model is not None:
        nlp = spacy.load(output_dir)
        print("Loaded model '%s'", model)
    else:
        nlp = spacy.blank("en")
        print("Created Blank en model")
        
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last = True)
    else:
        ner = nlp.get_pipe("ner")
        
    for _,annotations in training_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
            
    other_pipes = [x for x in nlp.pipe_names if x != 'ner']
    with nlp.disable_pipes(*other_pipes):
        if model is None:
            nlp.begin_training()
        else:
            nlp.resume_training()
            
        for itn in tqdm(range(n_iter)):
            random.shuffle(training_data)
            batches = minibatch(training_data, size = compounding(4.0, 500.0, 1.001))
            losses = {}
            for batch in batches:
                text, annotations = zip(*batch)
                nlp.update(
                    text,
                    annotations,
                    drop = 0.5,
                    losses = losses
                )
            print("losses : ", losses)


## === cell 11
sentiment = 'positive'

training_data = get_training_data(sentiment)
model_path = get_model_out_path(sentiment)

train(training_data, model_path, n_iter = 2, model = None)

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/111860517.py in <cell line: 0>()
      4 model_path = get_model_out_path(sentiment)
      5 
----> 6 train(training_data, model_path, n_iter = 2, model = None)

/tmp/ipykernel_11/4223524145.py in train(training_data, output_dir, n_iter, model)
      9     if 'ner' not in nlp.pipe_names:
     10         ner = nlp.create_pipe('ner')
---> 11         nlp.add_pipe(ner, last = True)
     12     else:
     13         ner = nlp.get_pipe("ner")

/usr/local/lib/python3.11/dist-packages/spacy/language.py in add_pipe(self, factory_name, name, before, after, first, last, source, config, raw_config, validate)
    809             bad_val = repr(factory_name)
    810             err = Errors.E966.format(component=bad_val, name=name)
--> 811             raise ValueError(err)
    812         name = name if name is not None else factory_name
    813         if name in self.component_names:

ValueError: [E966] `nlp.add_pipe` now takes the string name of the registered component factory, not a callable component. Expected string, but got <spacy.pipeline.ner.EntityRecognizer object at 0x7ffec6fa61f0> (name: 'None').

- If you created your component with `nlp.create_pipe('name')`: remove nlp.create_pipe and call `nlp.add_pipe('name')` instead.

- If you passed in a component like `TextCategorizer()`: call `nlp.add_pipe` with the string name instead, e.g. `nlp.add_pipe('textcat')`.

- If you're using a custom component: Add the decorator `@Language.component` (for function components) or `@Language.factory` (for class components / factories) to your custom component and assign it a name, e.g. `@Language.component('your_name')`. You can then run `nlp.add_pipe('your_name')` to add it to the pipeline.

## === cell 12
sentiment = 'negative'

training_data = get_training_data(sentiment)
model_path = get_model_out_path(sentiment)

train(training_data, model_path, n_iter = 2, model = None)

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2814629455.py in <cell line: 0>()
      4 model_path = get_model_out_path(sentiment)
      5 
----> 6 train(training_data, model_path, n_iter = 2, model = None)

/tmp/ipykernel_11/4223524145.py in train(training_data, output_dir, n_iter, model)
      9     if 'ner' not in nlp.pipe_names:
     10         ner = nlp.create_pipe('ner')
---> 11         nlp.add_pipe(ner, last = True)
     12     else:
     13         ner = nlp.get_pipe("ner")

/usr/local/lib/python3.11/dist-packages/spacy/language.py in add_pipe(self, factory_name, name, before, after, first, last, source, config, raw_config, validate)
    809             bad_val = repr(factory_name)
    810             err = Errors.E966.format(component=bad_val, name=name)
--> 811             raise ValueError(err)
    812         name = name if name is not None else factory_name
    813         if name in self.component_names:

ValueError: [E966] `nlp.add_pipe` now takes the string name of the registered component factory, not a callable component. Expected string, but got <spacy.pipeline.ner.EntityRecognizer object at 0x7ffec6fa4740> (name: 'None').

- If you created your component with `nlp.create_pipe('name')`: remove nlp.create_pipe and call `nlp.add_pipe('name')` instead.

- If you passed in a component like `TextCategorizer()`: call `nlp.add_pipe` with the string name instead, e.g. `nlp.add_pipe('textcat')`.

- If you're using a custom component: Add the decorator `@Language.component` (for function components) or `@Language.factory` (for class components / factories) to your custom component and assign it a name, e.g. `@Language.component('your_name')`. You can then run `nlp.add_pipe('your_name')` to add it to the pipeline.

## === cell 13
sentiment = 'neutral'

training_data = get_training_data(sentiment)
model_path = get_model_out_path(sentiment)

train(training_data, model_path, n_iter = 2, model = None)

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4284250764.py in <cell line: 0>()
      4 model_path = get_model_out_path(sentiment)
      5 
----> 6 train(training_data, model_path, n_iter = 2, model = None)

/tmp/ipykernel_11/4223524145.py in train(training_data, output_dir, n_iter, model)
      9     if 'ner' not in nlp.pipe_names:
     10         ner = nlp.create_pipe('ner')
---> 11         nlp.add_pipe(ner, last = True)
     12     else:
     13         ner = nlp.get_pipe("ner")

/usr/local/lib/python3.11/dist-packages/spacy/language.py in add_pipe(self, factory_name, name, before, after, first, last, source, config, raw_config, validate)
    809             bad_val = repr(factory_name)
    810             err = Errors.E966.format(component=bad_val, name=name)
--> 811             raise ValueError(err)
    812         name = name if name is not None else factory_name
    813         if name in self.component_names:

ValueError: [E966] `nlp.add_pipe` now takes the string name of the registered component factory, not a callable component. Expected string, but got <spacy.pipeline.ner.EntityRecognizer object at 0x7ffec6fa5620> (name: 'None').

- If you created your component with `nlp.create_pipe('name')`: remove nlp.create_pipe and call `nlp.add_pipe('name')` instead.

- If you passed in a component like `TextCategorizer()`: call `nlp.add_pipe` with the string name instead, e.g. `nlp.add_pipe('textcat')`.

- If you're using a custom component: Add the decorator `@Language.component` (for function components) or `@Language.factory` (for class components / factories) to your custom component and assign it a name, e.g. `@Language.component('your_name')`. You can then run `nlp.add_pipe('your_name')` to add it to the pipeline.

## === cell 15
TRAINED_MODELS_BASE_PATH = '../input/tse-spacy-model/models/'

## === cell 16
def predict_entities(text, model):
    doc = model(text)
    ent_array = []
    for ent in doc.ents:
        start = text.find(ent.text)
        end = start + len(ent.text)
        new_int = [start, end, ent.label_]
        if new_int not in ent_array:
            ent_array.append([start, end, ent.label_])
    selected_text = text[ent_array[0][0] : ent_array[0][1]] if len(ent_array) > 1 else text
    return selected_text

## === cell 17
def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c))/(len(a) + len(b) - len(c))

## === cell 18
if TRAINED_MODELS_BASE_PATH is not None:
    print("Loading models from ", TRAINED_MODELS_BASE_PATH)
    model_pos = spacy.load(TRAINED_MODELS_BASE_PATH + 'model_pos')
    model_neg = spacy.load(TRAINED_MODELS_BASE_PATH + 'model_neg')
    model_neu = spacy.load(TRAINED_MODELS_BASE_PATH + 'model_neu')
    
    jaccard_score = 0
    for index, row in tqdm(train_data.iterrows(), total = train_data.shape[0]):
        text = row.text
        if row.sentiment == 'positive':
            jaccard_score += jaccard(predict_entities(text, model_pos), row.selected_text)
        elif row.sentiment == 'negative':
            jaccard_score += jaccard(predict_entities(text, model_neg), row.selected_text)
        else:
            jaccard_score += jaccard(predict_entities(text, model_neu), row.selected_text)    
    print(f'Average Jaccard Score is {jaccard_score / train_data.shape[0]}')

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/29770368.py in <cell line: 0>()
      1 if TRAINED_MODELS_BASE_PATH is not None:
      2     print("Loading models from ", TRAINED_MODELS_BASE_PATH)
----> 3     model_pos = spacy.load(TRAINED_MODELS_BASE_PATH + 'model_pos')
      4     model_neg = spacy.load(TRAINED_MODELS_BASE_PATH + 'model_neg')
      5     model_neu = spacy.load(TRAINED_MODELS_BASE_PATH + 'model_neu')

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

OSError: [E050] Can't find model '../input/tse-spacy-model/models/model_pos'. It doesn't seem to be a Python package or a valid path to a data directory.

## === cell 20
if TRAINED_MODELS_BASE_PATH is not None:
    print("Loading models from ", TRAINED_MODELS_BASE_PATH)
    model_pos = spacy.load(TRAINED_MODELS_BASE_PATH + 'model_pos')
    model_neg = spacy.load(TRAINED_MODELS_BASE_PATH + 'model_neg')
    model_neu = spacy.load(TRAINED_MODELS_BASE_PATH + 'model_neu')
    
    final_data = []
    for index, row in tqdm(test.iterrows(), total = test.shape[0]):
        text = row.text
        if row.sentiment == 'positive':
            final_data.append(predict_entities(text, model_pos))
        elif row.sentiment == 'negative':
            final_data.append(predict_entities(text, model_neg))
        else:
            final_data.append(predict_entities(text, model_neu))

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/3189345949.py in <cell line: 0>()
      1 if TRAINED_MODELS_BASE_PATH is not None:
      2     print("Loading models from ", TRAINED_MODELS_BASE_PATH)
----> 3     model_pos = spacy.load(TRAINED_MODELS_BASE_PATH + 'model_pos')
      4     model_neg = spacy.load(TRAINED_MODELS_BASE_PATH + 'model_neg')
      5     model_neu = spacy.load(TRAINED_MODELS_BASE_PATH + 'model_neu')

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

OSError: [E050] Can't find model '../input/tse-spacy-model/models/model_pos'. It doesn't seem to be a Python package or a valid path to a data directory.

## === cell 21
testID = test['textID']

## === cell 22
df = pd.DataFrame(list(zip(testID, final_data)), columns = ['textID', 'selected_text'])

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4193662018.py in <cell line: 0>()
----> 1 df = pd.DataFrame(list(zip(testID, final_data)), columns = ['textID', 'selected_text'])

NameError: name 'final_data' is not defined

## === cell 23
df.head()

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/964094849.py in <cell line: 0>()
----> 1 df.head()

NameError: name 'df' is not defined

## === cell 24
df.to_csv("submission.csv", index=False)
print("successfully saved")

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2873711861.py in <cell line: 0>()
----> 1 df.to_csv("submission.csv", index=False)
      2 print("successfully saved")

NameError: name 'df' is not defined
