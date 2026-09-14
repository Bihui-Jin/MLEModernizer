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
plotly==5.24.1
plotly-express==0.4.1
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

0.6579

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train_df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
train_df.head(5)


## === cell 2
train_df.info()


## === cell 3
train_df.dropna(inplace=True)


## === cell 4
def jacquard_f(text1, text2):
    text1 = set(text1.lower().split())
    text2 = set(text2.lower().split())
    inter = text1.intersection(text2)
    return len(inter)/(len(text1) + len(text2) - len(inter))
    


## === cell 5
jacquard_values = [] 
for ind, row in train_df.iterrows():
    s1 = row.text
    s2 = row.selected_text
    jacquard_values.append([s1, s2, jacquard_f(s1, s2)])
jacquard = pd.DataFrame(jacquard_values, columns=["text","selected_text","jac"])
train_df = train_df.merge(jacquard, how="outer",on="text")
train_df.head(3)


## === cell 6
import matplotlib.pyplot as plt
import seaborn as sns

p1=sns.kdeplot(train_df[train_df['sentiment']=='positive']['jac'], shade=True, color="r")
p2=sns.kdeplot(train_df[train_df['sentiment']=='negative']['jac'], shade=True, color="b")
p3=sns.kdeplot(train_df[train_df['sentiment']=='neutral']['jac'], shade=True, color="g")


## === cell 7
train_df['num_words_text']= train_df['text'].apply(lambda x: len(str(x).split()))


## === cell 9
less_three = train_df[train_df['num_words_text']<=2]
less_three.groupby('sentiment').mean()['jac']
less_three.head(5)


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in _agg_py_fallback(self, how, values, ndim, alt)
   1941         try:
-> 1942             res_values = self._grouper.agg_series(ser, alt, preserve_dtype=True)
   1943         except Exception as err:

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/ops.py in agg_series(self, obj, func, preserve_dtype)
    863 
--> 864         result = self._aggregate_series_pure_python(obj, func)
    865 

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/ops.py in _aggregate_series_pure_python(self, obj, func)
    884         for i, group in enumerate(splitter):
--> 885             res = func(group)
    886             res = extract_result(res)

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in <lambda>(x)
   2453                 "mean",
-> 2454                 alt=lambda x: Series(x, copy=False).mean(numeric_only=numeric_only),
   2455                 numeric_only=numeric_only,

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in mean(self, axis, skipna, numeric_only, **kwargs)
   6548     ):
-> 6549         return NDFrame.mean(self, axis, skipna, numeric_only, **kwargs)
   6550 

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in mean(self, axis, skipna, numeric_only, **kwargs)
  12419     ) -> Series | float:
> 12420         return self._stat_function(
  12421             "mean", nanops.nanmean, axis, skipna, numeric_only, **kwargs

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _stat_function(self, name, func, axis, skipna, numeric_only, **kwargs)
  12376 
> 12377         return self._reduce(
  12378             func, name=name, axis=axis, skipna=skipna, numeric_only=numeric_only

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in _reduce(self, op, name, axis, skipna, numeric_only, filter_type, **kwds)
   6456                 )
-> 6457             return op(delegate, skipna=skipna, **kwds)
   6458 

/usr/local/lib/python3.11/dist-packages/pandas/core/nanops.py in f(values, axis, skipna, **kwds)
    146             else:
--> 147                 result = alt(values, axis=axis, skipna=skipna, **kwds)
    148 

/usr/local/lib/python3.11/dist-packages/pandas/core/nanops.py in new_func(values, axis, skipna, mask, **kwargs)
    403 
--> 404         result = func(values, axis=axis, skipna=skipna, mask=mask, **kwargs)
    405 

/usr/local/lib/python3.11/dist-packages/pandas/core/nanops.py in nanmean(values, axis, skipna, mask)
    719     the_sum = values.sum(axis, dtype=dtype_sum)
--> 720     the_sum = _ensure_numeric(the_sum)
    721 

/usr/local/lib/python3.11/dist-packages/pandas/core/nanops.py in _ensure_numeric(x)
   1700             # GH#44008, GH#36703 avoid casting e.g. strings to numeric
-> 1701             raise TypeError(f"Could not convert string '{x}' to numeric")
   1702         try:

TypeError: Could not convert string '852edc3769cdeaa93be48698d535f2bc361e245ebed4306d9bb732cd66413b921ccd0bde99f084945b3f3b08099bd1d86c69f6f69e5073dcf1e95b1c02ea061b1d834196cadffdde144991a663e7c602ae95d29a4caaf786894bd8d909d39141e2147763c0da2b190b25329e7d965eb0e374b2ec1c0bed85cdeb258b89d2d4deeb43c6ca216383cfedf94a5384eeb1fdcde08f8b684dd1880523269ff15d5ccf8f74c4f71cfbc22efc4d870375e4c9b93e52800bbae54143779810abc7071cafbadd63ec08321ce1f25996c236234a6c8c35bc879a25b00404648e1c44e6dbd2ce706e150b70a52b69d7e1d4c6184863f7e258097aa8734230b660c947090b7e4ed52c4a1559c3c50f777182cef9a0475aee7f6ecaa33c8a02aacdcb65a796ba1be05ef2cf68b1b9c95f65f8001e034570086deb67c566a3b2a036930b5a21be4f566959d3771b845b380af1aa85433e4a49b4ccb6f6bd82c0d18c11a2bf69c690e2a28bf982f68664181f4097327a15d6690167c461c6a7876c66da94853572900c6ea9affd97afa4ae072385a6b7ea317db125d64213f65406f0460d611d6dd07105c90e2f13043ee2708719cc9f3be484afc8bcfff762bbbe6803f71f7ac5ca5e8078049b721fdeebda41433cb03f2aa6ad5df0ca9b26804c67' to numeric

The above exception was the direct cause of the following exception:

TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2824255677.py in <cell line: 0>()
      1 less_three = train_df[train_df['num_words_text']<=2]
----> 2 less_three.groupby('sentiment').mean()['jac']
      3 less_three.head(5)

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in mean(self, numeric_only, engine, engine_kwargs)
   2450             )
   2451         else:
-> 2452             result = self._cython_agg_general(
   2453                 "mean",
   2454                 alt=lambda x: Series(x, copy=False).mean(numeric_only=numeric_only),

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in _cython_agg_general(self, how, alt, numeric_only, min_count, **kwargs)
   1996             return result
   1997 
-> 1998         new_mgr = data.grouped_reduce(array_func)
   1999         res = self._wrap_agged_manager(new_mgr)
   2000         if how in ["idxmin", "idxmax"]:

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in grouped_reduce(self, func)
   1467                 #  while others do not.
   1468                 for sb in blk._split():
-> 1469                     applied = sb.apply(func)
   1470                     result_blocks = extend_blocks(applied, result_blocks)
   1471             else:

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/blocks.py in apply(self, func, **kwargs)
    391         one
    392         """
--> 393         result = func(self.values, **kwargs)
    394 
    395         result = maybe_coerce_values(result)

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in array_func(values)
   1993 
   1994             assert alt is not None
-> 1995             result = self._agg_py_fallback(how, values, ndim=data.ndim, alt=alt)
   1996             return result
   1997 

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in _agg_py_fallback(self, how, values, ndim, alt)
   1944             msg = f"agg function failed [how->{how},dtype->{ser.dtype}]"
   1945             # preserve the kind of exception that raised
-> 1946             raise type(err)(msg) from err
   1947 
   1948         if ser.dtype == object:

TypeError: agg function failed [how->mean,dtype->object]

## === cell 10
from nltk.corpus import stopwords
stopword = stopwords.words('english')


## === cell 11
import re
import string
def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = text.split()
    words = [t for t in text if t not in stopword]
    return words
train_df['list_words'] = train_df['text'].apply(lambda x:clean_text(x))
train_df['list_words_selected'] = train_df['selected_text_x'].apply(lambda x:clean_text(x))


## === cell 13
train_df['list_words'].head(3)


## === cell 14
from collections import Counter
top_words_text = Counter([item for sublist in train_df['list_words'] for item in sublist])
top_words_selected_text = Counter([item for sublist in train_df['list_words_selected'] for item in sublist])


## === cell 17
positives = train_df[train_df['sentiment']=='positive']
negatives = train_df[train_df['sentiment']=='negative']
neutrals = train_df[train_df['sentiment']=='neutral']


## === cell 18
top = Counter([item for sublist in positives['list_words'] for item in sublist])
temp_positive = pd.DataFrame(top.most_common(20))
temp_positive.columns = ['Common_words','count']
temp_positive


## === cell 19
top = Counter([item for sublist in negatives['list_words'] for item in sublist])
temp_negative = pd.DataFrame(top.most_common(20))
temp_negative.columns = ['Common_words','count']
temp_negative


## === cell 20
top = Counter([item for sublist in neutrals['list_words'] for item in sublist])
temp_neutral = pd.DataFrame(top.most_common(20))
temp_neutral.columns = ['Common_words','count']
temp_neutral


## === cell 21
import plotly.express as px
fig = px.treemap(temp_positive, path=['Common_words'], values='count',title='Common Postive Words')
fig.show()
fig = px.treemap(temp_negative, path=['Common_words'], values='count',title='Common Negative Words')
fig.show()
fig = px.treemap(temp_neutral, path=['Common_words'], values='count',title='Common Neutral Words')
fig.show()


## === cell 22
def get_unique_words(sentiment,numwords,raw_words):
    other_words = []
    for item in train_df[train_df.sentiment != sentiment]['list_words']:
        for word in item:
            other_words.append(word)
    other_words= list(set(other_words))
    category_words = [x for x in raw_words if x not in other_words]
    newcounter = Counter()
    for item in train_df[train_df.sentiment == sentiment]['list_words']:
        for word in item:
            newcounter[word] += 1
    keep = list(category_words)
    for word in list(newcounter):
        if word not in keep:
            del newcounter[word]
    unique_words = pd.DataFrame(newcounter.most_common(numwords), columns = ['words','count'])
    return unique_words


## === cell 23
raw_text = [word for word_list in train_df['list_words'] for word in word_list]
unique_positive= get_unique_words('positive', 20, raw_text)
unique_negative= get_unique_words('negative', 20, raw_text)
unique_neutral= get_unique_words('neutral', 20, raw_text)
unique_positive


## === cell 24
train_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
test_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
train_df['text_length'] = train_df['text'].apply(lambda x:len(str(x).split())) 
train_df = train_df[train_df['text_length']>=3]


## === cell 25
def format_data(sentiment):
    formatted_data = []
    for index, row in train_df.iterrows():
        if row.sentiment == sentiment:
            selected_text = row.selected_text
            text = row.text
            start = text.find(selected_text)
            end = start + len(selected_text)
            formatted_data.append((text, {"entities": [[start, end, 'selected_text']]}))
    return formatted_data


## === cell 26
def train(train_data, output_path, n_iter=20, model=None):
    if model is not None:
        nlp = spacy.load(output_path) 
    else:
        nlp = spacy.blank("en")
    
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
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
    nlp.meta["name"] = "st_ner"
    nlp.to_disk(output_path)


## === cell 27
import spacy
from tqdm import tqdm
import random
from spacy.util import minibatch, compounding
sentiment = 'positive'
train_data_positive = format_data(sentiment)
train(train_data_positive, 'positive', n_iter=3, model=None)


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2096756029.py in <cell line: 0>()
      5 sentiment = 'positive'
      6 train_data_positive = format_data(sentiment)
----> 7 train(train_data_positive, 'positive', n_iter=3, model=None)

/tmp/ipykernel_11/3032447080.py in train(train_data, output_path, n_iter, model)
      7     if "ner" not in nlp.pipe_names:
      8         ner = nlp.create_pipe("ner")
----> 9         nlp.add_pipe(ner, last=True)
     10     else:
     11         ner = nlp.get_pipe("ner")

/usr/local/lib/python3.11/dist-packages/spacy/language.py in add_pipe(self, factory_name, name, before, after, first, last, source, config, raw_config, validate)
    809             bad_val = repr(factory_name)
    810             err = Errors.E966.format(component=bad_val, name=name)
--> 811             raise ValueError(err)
    812         name = name if name is not None else factory_name
    813         if name in self.component_names:

ValueError: [E966] `nlp.add_pipe` now takes the string name of the registered component factory, not a callable component. Expected string, but got <spacy.pipeline.ner.EntityRecognizer object at 0x7ffeb1cdd4d0> (name: 'None').

- If you created your component with `nlp.create_pipe('name')`: remove nlp.create_pipe and call `nlp.add_pipe('name')` instead.

- If you passed in a component like `TextCategorizer()`: call `nlp.add_pipe` with the string name instead, e.g. `nlp.add_pipe('textcat')`.

- If you're using a custom component: Add the decorator `@Language.component` (for function components) or `@Language.factory` (for class components / factories) to your custom component and assign it a name, e.g. `@Language.component('your_name')`. You can then run `nlp.add_pipe('your_name')` to add it to the pipeline.

## === cell 28
sentiment = 'negative'
train_data_negative = format_data(sentiment)
train(train_data_negative, 'negative', n_iter=3, model=None)


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2822205976.py in <cell line: 0>()
      1 sentiment = 'negative'
      2 train_data_negative = format_data(sentiment)
----> 3 train(train_data_negative, 'negative', n_iter=3, model=None)

/tmp/ipykernel_11/3032447080.py in train(train_data, output_path, n_iter, model)
      7     if "ner" not in nlp.pipe_names:
      8         ner = nlp.create_pipe("ner")
----> 9         nlp.add_pipe(ner, last=True)
     10     else:
     11         ner = nlp.get_pipe("ner")

/usr/local/lib/python3.11/dist-packages/spacy/language.py in add_pipe(self, factory_name, name, before, after, first, last, source, config, raw_config, validate)
    809             bad_val = repr(factory_name)
    810             err = Errors.E966.format(component=bad_val, name=name)
--> 811             raise ValueError(err)
    812         name = name if name is not None else factory_name
    813         if name in self.component_names:

ValueError: [E966] `nlp.add_pipe` now takes the string name of the registered component factory, not a callable component. Expected string, but got <spacy.pipeline.ner.EntityRecognizer object at 0x7ffeb1cdf3e0> (name: 'None').

- If you created your component with `nlp.create_pipe('name')`: remove nlp.create_pipe and call `nlp.add_pipe('name')` instead.

- If you passed in a component like `TextCategorizer()`: call `nlp.add_pipe` with the string name instead, e.g. `nlp.add_pipe('textcat')`.

- If you're using a custom component: Add the decorator `@Language.component` (for function components) or `@Language.factory` (for class components / factories) to your custom component and assign it a name, e.g. `@Language.component('your_name')`. You can then run `nlp.add_pipe('your_name')` to add it to the pipeline.

## === cell 29
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


## === cell 30
predicted_selected_text = []
model_pos = spacy.load('positive')
model_neg = spacy.load('negative')
        
for index, row in test_df.iterrows():
    text = row.text
    output_str = ""
    if row.sentiment == 'neutral' or len(text.split()) <= 2:
        predicted_selected_text.append(text)
    elif row.sentiment == 'positive':
        predicted_selected_text.append(predict_entities(text, model_pos))
    else:
        predicted_selected_text.append(predict_entities(text, model_neg))
        
test_df['selected_text'] = predicted_selected_text


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/3772522067.py in <cell line: 0>()
      1 predicted_selected_text = []
----> 2 model_pos = spacy.load('positive')
      3 model_neg = spacy.load('negative')
      4 
      5 for index, row in test_df.iterrows():

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

OSError: [E050] Can't find model 'positive'. It doesn't seem to be a Python package or a valid path to a data directory.

## === cell 31
submission_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
submission_df['selected_text'] = test_df['selected_text']
submission_df.to_csv("submission.csv", index=False)
submission_df.head(10)


## --- ERROR in cell 31, traceback:
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
/tmp/ipykernel_11/4140973978.py in <cell line: 0>()
      1 submission_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
----> 2 submission_df['selected_text'] = test_df['selected_text']
      3 submission_df.to_csv("submission.csv", index=False)
      4 submission_df.head(10)

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
