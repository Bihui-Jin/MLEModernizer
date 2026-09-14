# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

Higher is better.

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
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 2
train = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
sample = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")



## === cell 3
train.dropna(inplace=True)



## === cell 4
train_data = train.copy()




## === cell 5
def get_training_data(sentiment):
    training_data = []
    for data in train_data.values:
        if data[3] == sentiment:
            text = data[1]
            selected_text = data[2]
            start = text.find(selected_text)
            end = start + len(selected_text)
            training_data.append((text, {"entities": [[start, end, "selected_text"]]}))
    return training_data




## === cell 6
def get_model_out_path(sentiment):
    if sentiment == "positive":
        return "model_pos"
    elif sentiment == "negative":
        return "model_neg"
    else:
        return "model_neu"




## === cell 7
def save_model(output_dir, nlp, new_model_name):
    output_path = os.path.join("/kaggle/working", output_dir)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    nlp.meta["name"] = new_model_name
    nlp.to_disk(output_path)
    print("Saved model to", output_path)
    return output_path




## === cell 8
def train(training_data, output_dir, n_iter=10, drop_rate=0.2):
    """
    Train a blank spaCy NER model.
    Reduced dropout (default 0.2) and a configurable number of iterations
    keep the core logic unchanged while giving the model more stable learning.
    """
    nlp = spacy.blank("en")
    print("Created Blank en model")
    if "ner" not in nlp.pipe_names:
        nlp.add_pipe("ner")
    ner = nlp.get_pipe("ner")
    for _, annotations in training_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
    other_pipes = [p for p in nlp.pipe_names if p != "ner"]
    with nlp.disable_pipes(*other_pipes):
        nlp.begin_training()
        for itn in tqdm(range(n_iter), desc="Training"):
            random.shuffle(training_data)
            batches = minibatch(training_data, size=compounding(4.0, 500.0, 1.001))
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,
                    annotations,
                    drop=drop_rate,
                    losses=losses,
                )
            print(f"Iteration {itn+1} losses:", losses)
    saved_path = save_model(output_dir, nlp, f"{output_dir}_nlp")
    return nlp, saved_path




## === cell 9
sentiment = "positive"
training_data_pos = get_training_data(sentiment)
model_pos_path = get_model_out_path(sentiment)
model_pos, _ = train(training_data_pos, model_pos_path, n_iter=20)



## === cell 10
sentiment = "negative"
training_data_neg = get_training_data(sentiment)
model_neg_path = get_model_out_path(sentiment)
model_neg, _ = train(training_data_neg, model_neg_path, n_iter=20)



## === cell 11
sentiment = "neutral"
training_data_neu = get_training_data(sentiment)
model_neu_path = get_model_out_path(sentiment)
model_neu, _ = train(training_data_neu, model_neu_path, n_iter=20)




## === cell 12
def predict_entities(text, model):
    """
    Return the longest predicted entity (by character span).
    This aligns better with the required 'selected_text' and often improves Jaccard.
    """
    doc = model(text)
    ent_array = []
    for ent in doc.ents:
        start = text.find(ent.text)
        end = start + len(ent.text)
        entry = [start, end, ent.label_]
        if entry not in ent_array:
            ent_array.append(entry)
    if len(ent_array) > 0:
        best = max(ent_array, key=lambda x: x[1] - x[0])
        sel = text[best[0] : best[1]]
    else:
        sel = text
    return sel




## === cell 13
def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))




## === cell 14
jaccard_score = 0.0
for _, row in tqdm(train_data.iterrows(), total=train_data.shape[0], desc="Eval"):
    txt = row.text
    if row.sentiment == "positive":
        pred = predict_entities(txt, model_pos)
    elif row.sentiment == "negative":
        pred = predict_entities(txt, model_neg)
    else:
        pred = predict_entities(txt, model_neu)
    jaccard_score += jaccard(pred, row.selected_text)
print(
    f"Average Jaccard Score on train split: {jaccard_score / train_data.shape[0]:.5f}"
)



## === cell 15
final_data = []
for _, row in tqdm(test.iterrows(), total=test.shape[0], desc="Predict"):
    txt = row.text
    if row.sentiment == "positive":
        final_data.append(predict_entities(txt, model_pos))
    elif row.sentiment == "negative":
        final_data.append(predict_entities(txt, model_neg))
    else:
        final_data.append(predict_entities(txt, model_neu))



## === cell 16
testID = test["textID"]
df = pd.DataFrame(list(zip(testID, final_data)), columns=["textID", "selected_text"])



## === cell 17
df.head()



## === cell 18
df.to_csv("submission.csv", index=False)
print("successfully saved submission.csv")
