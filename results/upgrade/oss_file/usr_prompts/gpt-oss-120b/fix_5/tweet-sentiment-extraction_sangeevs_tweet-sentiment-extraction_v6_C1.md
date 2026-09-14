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

# 5. Code solution

## === cell 0
import pandas as pd
import numpy as np
import re
import string
import random
from collections import Counter
import nltk
import spacy
from tqdm import tqdm
from spacy.util import minibatch, compounding
from spacy.training.example import Example



## === cell 1
train_df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
test_df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
train_df["text_length"] = train_df["text"].apply(lambda x: len(str(x).split()))
train_df = train_df[train_df["text_length"] >= 3]



## === cell 2
train_df.info()



## === cell 3
train_df.dropna(inplace=True)




## === cell 4
def jacquard_f(text1, text2):
    text1 = set(text1.lower().split())
    text2 = set(text2.lower().split())
    inter = text1.intersection(text2)
    return len(inter) / (len(text1) + len(text2) - len(inter))




## === cell 5
train_df["jac"] = train_df.apply(
    lambda row: jacquard_f(row["text"], row["selected_text"]), axis=1
)
train_df.head(3)



## === cell 6
pass



## === cell 7
train_df["num_words_text"] = train_df["text"].apply(lambda x: len(str(x).split()))



## === cell 8
less_three = train_df[train_df["num_words_text"] <= 2]
mean_jac = less_three.groupby("sentiment")["jac"].mean()
print("Mean Jaccard for very short texts per sentiment:")
print(mean_jac)



## === cell 9
nltk.download("stopwords")
from nltk.corpus import stopwords

stopword = stopwords.words("english")




## === cell 10
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\w*\d\w*", "", text)
    tokens = text.split()
    words = [t for t in tokens if t not in stopword]
    return words


train_df["list_words"] = train_df["text"].apply(lambda x: clean_text(x))
train_df["list_words_selected"] = train_df["selected_text"].apply(
    lambda x: clean_text(x)
)



## === cell 11
train_df["list_words"].head(3)



## === cell 12
top_words_text = Counter(
    [item for sublist in train_df["list_words"] for item in sublist]
)
top_words_selected_text = Counter(
    [item for sublist in train_df["list_words_selected"] for item in sublist]
)



## === cell 13
positives = train_df[train_df["sentiment"] == "positive"]
negatives = train_df[train_df["sentiment"] == "negative"]
neutrals = train_df[train_df["sentiment"] == "neutral"]



## === cell 14
top = Counter([item for sublist in positives["list_words"] for item in sublist])
temp_positive = pd.DataFrame(top.most_common(20))
temp_positive.columns = ["Common_words", "count"]
temp_positive



## === cell 15
top = Counter([item for sublist in negatives["list_words"] for item in sublist])
temp_negative = pd.DataFrame(top.most_common(20))
temp_negative.columns = ["Common_words", "count"]
temp_negative



## === cell 16
top = Counter([item for sublist in neutrals["list_words"] for item in sublist])
temp_neutral = pd.DataFrame(top.most_common(20))
temp_neutral.columns = ["Common_words", "count"]
temp_neutral



## === cell 17
pass




## === cell 18
def get_unique_words(sentiment, numwords, raw_words):
    other_words = set(
        word
        for s, lst in train_df[train_df.sentiment != sentiment][
            ["sentiment", "list_words"]
        ].itertuples(index=False)
        for word in lst
    )
    category_words = [x for x in raw_words if x not in other_words]
    newcounter = Counter()
    for lst in train_df[train_df.sentiment == sentiment]["list_words"]:
        for word in lst:
            newcounter[word] += 1
    for word in list(newcounter):
        if word not in category_words:
            del newcounter[word]
    unique_words = pd.DataFrame(
        newcounter.most_common(numwords), columns=["words", "count"]
    )
    return unique_words




## === cell 19
raw_text = [word for word_list in train_df["list_words"] for word in word_list]
unique_positive = get_unique_words("positive", 20, raw_text)
unique_negative = get_unique_words("negative", 20, raw_text)
unique_neutral = get_unique_words("neutral", 20, raw_text)
unique_positive



## === cell 20
pass




## === cell 21
def format_data(sentiment):
    formatted_data = []
    for _, row in train_df.iterrows():
        if row.sentiment == sentiment:
            selected_text = row.selected_text
            text = row.text
            start = text.find(selected_text)
            end = start + len(selected_text)
            formatted_data.append((text, {"entities": [[start, end, "selected_text"]]}))
    return formatted_data




## === cell 22
def train(train_data, output_path, n_iter=5, model=None):
    """
    Train a spaCy NER model for the given sentiment using the spaCy v3 API.
    """
    nlp = spacy.blank("en")

    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    def get_examples():
        for text, ann in train_data:
            doc = nlp.make_doc(text)
            yield Example.from_dict(doc, ann)

    optimizer = nlp.initialize(get_examples)

    for itn in range(n_iter):
        random.shuffle(train_data)
        batches = minibatch(train_data, size=compounding(4.0, 500.0, 1.001))
        losses = {}
        for batch in batches:
            texts, annotations = zip(*batch)
            examples = [
                Example.from_dict(nlp.make_doc(t), a)
                for t, a in zip(texts, annotations)
            ]
            nlp.update(
                examples,
                drop=0.5,
                losses=losses,
                sgd=optimizer,
            )
    nlp.to_disk(output_path)




## === cell 23
sentiment = "positive"
train_data_positive = format_data(sentiment)
train(train_data_positive, "positive_model", n_iter=5)

sentiment = "negative"
train_data_negative = format_data(sentiment)
train(train_data_negative, "negative_model", n_iter=5)




## === cell 24
def predict_entities(text, model):
    doc = model(text)
    ent_array = []
    for ent in doc.ents:
        start = text.find(ent.text)
        end = start + len(ent.text)
        new_int = [start, end, ent.label_]
        if new_int not in ent_array:
            ent_array.append(new_int)
    selected_text = text[ent_array[0][0] : ent_array[0][1]] if ent_array else text
    return selected_text




## === cell 25
model_pos = spacy.load("positive_model")
model_neg = spacy.load("negative_model")

predicted_selected_text = []
for _, row in test_df.iterrows():
    text = row.text
    if row.sentiment == "neutral" or len(text.split()) <= 2:
        predicted_selected_text.append(text)
    elif row.sentiment == "positive":
        predicted_selected_text.append(predict_entities(text, model_pos))
    else:  # negative
        predicted_selected_text.append(predict_entities(text, model_neg))

test_df["selected_text"] = predicted_selected_text



## === cell 26
submission_df = pd.DataFrame(
    {"textID": test_df["textID"], "selected_text": test_df["selected_text"]}
)
submission_df.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
submission_df.head(10)
