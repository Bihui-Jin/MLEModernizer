# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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

import spacy

spacy.require_gpu()

tqdm_notebook.pandas()




## === cell 1
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("There are %d GPU(s) available." % torch.cuda.device_count())
    print("We will use the GPU:", torch.cuda.get_device_name(0))
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")




## === cell 2
try:
    df_train = pd.read_csv("data/train.csv")
    df_test = pd.read_csv("data/test.csv")
except:
    df_train = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
    df_test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")




## === cell 3
df_train["text_original"] = df_train["text"]




## === cell 4
df_train.fillna("", inplace=True)
df_test.fillna("", inplace=True)




## === cell 5
lb = LabelEncoder()
df_train["target"] = lb.fit_transform(df_train["sentiment"])




## === cell 6
tf = TfidfVectorizer(ngram_range=(1, 1))
X_train, X_val, y_train, y_val = train_test_split(
    tf.fit_transform(df_train["text"]), df_train["target"]
)
df_train["is_training"] = [1 if x in y_train.index else 0 for x in df_train.index]




## === cell 7
parameters = {}
clf = OneVsRestClassifier(LogisticRegression(solver="lbfgs"))
clf.fit(X_train, y_train)




## === cell 8
y_val_predict_sentiment = clf.predict(X_val)




## === cell 9
f1_score(y_val, y_val_predict_sentiment, average="weighted")




## === cell 10
train_data = []
for idx, row in df_train[(df_train["sentiment"] != "neutral")].iterrows():
    text = row["text"]
    selected_text = row["selected_text"]
    if selected_text in text:
        entities = []
        try:
            for match in re.finditer(re.escape(selected_text), text):
                start_char = match.start()
                end_char = match.end()
                entity_label = row["sentiment"]
                entities.append((start_char, end_char, entity_label))
        except Exception as e:
            print(text)
            print(selected_text)
            raise e
        train_data.append((text, {"entities": entities}))




## === cell 11
def spacy_train_custom(train_data, epochs=10):
    """Train a blank spaCy NER model using the current SpaCy v3 API.
    Optimized to create Example objects once, reducing per‑epoch overhead.
    """
    import spacy
    from spacy.training.example import Example
    from spacy.util import minibatch, compounding

    random.seed(42)

    nlp = spacy.blank("en")  # create a blank English model

    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    for _, annotations in train_data:
        for start, end, label in annotations.get("entities"):
            ner.add_label(label)

    optimizer = nlp.initialize()

    examples = [
        Example.from_dict(nlp.make_doc(text), annotation)
        for text, annotation in train_data
    ]

    for itn in range(epochs):
        random.shuffle(examples)
        losses = {}
        batches = minibatch(examples, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            nlp.update(
                batch,
                sgd=optimizer,
                drop=0.3,
                losses=losses,
            )
        print(f"Epoch {itn + 1}/{epochs} - Losses: {losses}")

    for text, _ in train_data[:2]:
        doc = nlp(text)
        print("Example:", text)
        print("Predicted entities:", [(ent.text, ent.label_) for ent in doc.ents])

    return nlp




## === cell 12
nlp = spacy_train_custom(train_data, epochs=30)




## === cell 13
from spacy import displacy

sample = df_train.sample().iloc[0]
doc = nlp(sample.text)
displacy.render(doc, style="ent")
print(sample.sentiment)
print(sample.selected_text)




## === cell 14
docs = list(nlp.pipe(df_test["text"], batch_size=64))
df_test["selected_text"] = [" ".join([ent.text for ent in doc.ents]) for doc in docs]

df_test["selected_text"] = [
    (
        row["text"]
        if row["sentiment"] == "neutral" or row["selected_text"] == ""
        else row["selected_text"]
    )
    for _, row in df_test.iterrows()
]




## === cell 15
df_test[["textID", "selected_text"]].to_csv("submission.csv", index=False)
