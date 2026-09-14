# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.8

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
print("Training Examples : {}".format(len(train)))
print("Testing Examples : {}".format(len(test)))


## === cell 2
train.head()


## === cell 3
import plotly.graph_objects as go
from plotly.subplots import make_subplots


## === cell 4
class_dist = train.groupby(["sentiment"]).count()['textID'].reset_index()
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {}]])
fig.add_trace(go.Pie(labels=class_dist.sentiment, values=class_dist.textID, name="% Distribution",hole=.5),
              1, 1)
fig.add_trace(go.Bar(x = class_dist.sentiment, y=class_dist.textID, name="Frequency Distribution"),
              1, 2)
fig.update_layout(title="% Distribution and Frequency Disstibution")


## === cell 5
import math
import re
from collections import Counter

WORD = re.compile(r"\w+")

def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)

def get_cosine(text1, text2):
    vec1 = text_to_vector(text1)
    vec2 = text_to_vector(text2)   
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

str1 = train.text[0]
str2 = train.selected_text[0]

print("String 1 : {}".format(str1))
print("String 2 : {}".format(str2))
cosine_score = get_cosine(str1,str2) 
print("Cosine Similarity of the above two sentences : {}%".format(np.round(cosine_score*100)))


## === cell 6
train['COSINE_Score'] = train.apply(lambda row:get_cosine(str(row['text']),str(row['selected_text'])),axis=1)
train.head()


## === cell 7
import plotly.express as px
fig = px.box(train,x='sentiment',y='COSINE_Score',color='sentiment')
fig.update_traces(quartilemethod="inclusive") # or "inclusive", or "linear" by default
fig.show()


## === cell 8
def save_model(output_dir, nlp, new_model_name):
    ''' This Function Saves model to 
    given output directory'''
    
    output_dir = f'../working/{output_dir}'
    if output_dir is not None:        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        nlp.meta["name"] = new_model_name
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)


## === cell 9


from tqdm import tqdm
import os
import nltk
import spacy
import random
from spacy.util import compounding
from spacy.util import minibatch

def train_model(train_data, output_dir, n_iter=20, model=None):
    """Load the model, set up the pipeline and train the entity recognizer."""
    ""
    if model is not None:
        nlp = spacy.load(output_dir)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")
    
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    else:
        ner = nlp.get_pipe("ner")
    
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        if model is None:
            nlp.begin_training()
        else:
            nlp.resume_training()


        for itn in tqdm(range(n_iter)):
            random.shuffle(train_data)
            batches = minibatch(train_data, size=compounding(4.0, 500.0, 1.001))    
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts,  # batch of texts
                            annotations,  # batch of annotations
                            drop=0.5,   # dropout - make it harder to memorise data
                            losses=losses, 
                            )
            print("Losses", losses)
    save_model(output_dir, nlp, 'st_ner')


## === cell 10
def get_training_data(df_train):
    '''
    Returns Trainong data in the format needed to train spacy NER
    '''
    train_data = []
    for index, row in df_train.iterrows():
        sentiment = row.sentiment #Store sentiment here
        selected_text = str(row.selected_text)
        text = str(row.text)
        start = text.find(selected_text)
        end = start + len(selected_text)
        train_data.append((text, {"entities": [[start, end, sentiment]]})) #sentiment as training
    return train_data


## === cell 11
model_path = '/models/model'
train_data = get_training_data(train)
train_model(train_data,model_path,n_iter=3,model=None)


## --- ERROR in cell 11, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/719727438.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0mmodel_path[0m [0;34m=[0m [0;34m'/models/model'[0m[0;34m[0m[0;34m[0m[0m
[1;32m      2[0m [0mtrain_data[0m [0;34m=[0m [0mget_training_data[0m[0;34m([0m[0mtrain[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 3[0;31m [0mtrain_model[0m[0;34m([0m[0mtrain_data[0m[0;34m,[0m[0mmodel_path[0m[0;34m,[0m[0mn_iter[0m[0;34m=[0m[0;36m3[0m[0;34m,[0m[0mmodel[0m[0;34m=[0m[0;32mNone[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/tmp/ipykernel_11/3979439492.py[0m in [0;36mtrain_model[0;34m(train_data, output_dir, n_iter, model)[0m
[1;32m     25[0m     [0;32mif[0m [0;34m"ner"[0m [0;32mnot[0m [0;32min[0m [0mnlp[0m[0;34m.[0m[0mpipe_names[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     26[0m         [0mner[0m [0;34m=[0m [0mnlp[0m[0;34m.[0m[0mcreate_pipe[0m[0;34m([0m[0;34m"ner"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 27[0;31m         [0mnlp[0m[0;34m.[0m[0madd_pipe[0m[0;34m([0m[0mner[0m[0;34m,[0m [0mlast[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     28[0m     [0;31m# otherwise, get it so we can add labels[0m[0;34m[0m[0;34m[0m[0m
[1;32m     29[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/spacy/language.py[0m in [0;36madd_pipe[0;34m(self, factory_name, name, before, after, first, last, source, config, raw_config, validate)[0m
[1;32m    809[0m             [0mbad_val[0m [0;34m=[0m [0mrepr[0m[0;34m([0m[0mfactory_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    810[0m             [0merr[0m [0;34m=[0m [0mErrors[0m[0;34m.[0m[0mE966[0m[0;34m.[0m[0mformat[0m[0;34m([0m[0mcomponent[0m[0;34m=[0m[0mbad_val[0m[0;34m,[0m [0mname[0m[0;34m=[0m[0mname[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 811[0;31m             [0;32mraise[0m [0mValueError[0m[0;34m([0m[0merr[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    812[0m         [0mname[0m [0;34m=[0m [0mname[0m [0;32mif[0m [0mname[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m [0;32melse[0m [0mfactory_name[0m[0;34m[0m[0;34m[0m[0m
[1;32m    813[0m         [0;32mif[0m [0mname[0m [0;32min[0m [0mself[0m[0;34m.[0m[0mcomponent_names[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: [E966] `nlp.add_pipe` now takes the string name of the registered component factory, not a callable component. Expected string, but got <spacy.pipeline.ner.EntityRecognizer object at 0x7fff82111620> (name: 'None').

- If you created your component with `nlp.create_pipe('name')`: remove nlp.create_pipe and call `nlp.add_pipe('name')` instead.

- If you passed in a component like `TextCategorizer()`: call `nlp.add_pipe` with the string name instead, e.g. `nlp.add_pipe('textcat')`.

- If you're using a custom component: Add the decorator `@Language.component` (for function components) or `@Language.factory` (for class components / factories) to your custom component and assign it a name, e.g. `@Language.component('your_name')`. You can then run `nlp.add_pipe('your_name')` to add it to the pipeline.

## === cell 12
def predict_entities(text, model):
    doc = model(text)
    ent_array = []
    for ent in doc.ents:
        start = text.find(ent.text)
        end = start + len(ent.text)
        new_int = [start, end, ent.label_]
        if new_int not in ent_array:
            ent_array.append([start, end, ent.label_])
    selected_text = text[ent_array[0][0]: ent_array[0][1]] if len(ent_array) > 0 else text
    return selected_text
