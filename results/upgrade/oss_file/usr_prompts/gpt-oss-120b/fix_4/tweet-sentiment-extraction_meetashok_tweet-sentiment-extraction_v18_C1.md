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

0.6536479592323303

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
from __future__ import unicode_literals, print_function
import pandas as pd
import os
import re
import string
from nltk.corpus import stopwords

import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
import datetime
from tqdm import tqdm


LABEL = "SELECTEDTEXT"


def read_data(datadir):
    train = pd.read_csv(os.path.join(datadir, "train.csv"))
    test = pd.read_csv(os.path.join(datadir, "test.csv"))
    sample_submission = pd.read_csv(os.path.join(datadir, "sample_submission.csv"))

    return (train, test, sample_submission)


def jaccard_similarity(string1, string2):
    wordset = lambda x: set(x.lower().split())
    a, b = wordset(string1), wordset(string2)

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))


def clean_text(text):
    """Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.
    Source: https://www.kaggle.com/tanulsingh077/twitter-sentiment-extaction-analysis-eda-and-model
    """
    text = str(text).lower()
    text = re.sub("\[.*?\]", "", text)  # remove words in square brackets
    text = re.sub("https?://\S+|www\.\S+", "", text)
    text = re.sub("<.*?>+", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub("\n", "", text)
    text = re.sub("\w*\d\w*", "", text)
    return text


def remove_stopwords(words):
    return [word for word in words if word not in stopwords.words("english")]


def training_data(dataframe):
    data = (
        dataframe.dropna()
        .assign(
            text=lambda x: x.apply(lambda x: x.text.lower(), axis=1),
            selected_text=lambda x: x.apply(lambda x: x.selected_text.lower(), axis=1),
        )
        .assign(
            start=lambda x: x.apply(lambda x: x.text.find(x.selected_text), axis=1),
            end=lambda x: x.apply(lambda x: x.start + len(x.selected_text), axis=1),
        )
    )

    positive = []
    negative = []

    for i, row in data.iterrows():
        if row.end > row.start:
            train_row = (row.text, {"entities": [(row.start, row.end, LABEL)]})

            if row.sentiment == "positive":
                positive.append(train_row)
            elif row.sentiment == "negative":
                negative.append(train_row)
            else:
                pass

    print(f"Positive data size: {len(positive):,}")
    print(f"Negative data size: {len(negative):,}")

    return positive, negative


def run_model(data, positivemodel, negativemodel, outputpath=None):
    print("Loading models...")
    if positivemodel is not None and os.path.isdir(positivemodel):
        positive_nlp = spacy.load(positivemodel)
    else:
        positive_nlp = None
        print(
            f"Positive model not found at '{positivemodel}'. Using fallback (full text)."
        )
    if negativemodel is not None and os.path.isdir(negativemodel):
        negative_nlp = spacy.load(negativemodel)
    else:
        negative_nlp = None
        print(
            f"Negative model not found at '{negativemodel}'. Using fallback (full text)."
        )

    data = data.dropna()

    textIDs, selected_texts = [], []
    jaccards = {"positive": [], "negative": [], "neutral": []}

    input_train = "selected_text" in data.columns

    print("Iterating data...")
    for _, row in tqdm(data.iterrows(), total=len(data)):
        text_lc = row.text.lower()
        if row.sentiment == "positive":
            if len(text_lc.split()) <= 2 or positive_nlp is None:
                selected_text = row.text
            else:
                ents = positive_nlp(text_lc).ents
                selected_text = ents[0].text if len(ents) > 0 else row.text
        elif row.sentiment == "negative":
            if len(text_lc.split()) <= 2 or negative_nlp is None:
                selected_text = row.text
            else:
                ents = negative_nlp(text_lc).ents
                selected_text = ents[0].text if len(ents) > 0 else row.text
        else:
            selected_text = row.text

        textIDs.append(row.textID)
        selected_texts.append(selected_text)

        if input_train:
            jaccard = jaccard_similarity(row.text, selected_text)
            jaccards[row.sentiment].append(jaccard)

    output_df = pd.DataFrame({"textID": textIDs, "selected_text": selected_texts})

    if not input_train:
        if not outputpath:
            suffix = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            outputpath = os.path.join("submissions", "submission_" + suffix + ".csv")
        os.makedirs(os.path.dirname(outputpath), exist_ok=True)

        print("Saving submission...")
        output_df.to_csv(outputpath, index=False)

    if input_train:
        nums, dens = [], []
        for key in ("positive", "negative", "neutral"):
            num = sum(jaccards[key])
            den = len(jaccards[key])
            if den > 0:
                print(f"Jaccard score for {key}: {num / den:.3f}")
                nums.append(num)
                dens.append(den)

        if sum(dens) > 0:
            print(f"Jaccard score for overall: {sum(nums) / sum(dens):.3f}")


def train_model(traindata, new_model_name, model=None, output_dir=None, n_iter=30):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    random.seed(0)
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    ner.add_label(LABEL)  # add new entity label to entity recognizer

    optimizer = nlp.initialize()

    move_names = list(ner.move_names)
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    with nlp.disable_pipes(*other_pipes):  # only train NER
        sizes = compounding(1.0, 4.0, 1.001)
        for itn in range(n_iter):
            random.shuffle(traindata)
            batches = minibatch(traindata, size=sizes)
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)
            print("Losses", losses)

    test_text = "i`d have responded, if i were going"
    doc = nlp(test_text)
    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)

    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        assert nlp2.get_pipe("ner").move_names == move_names
        doc2 = nlp2(test_text)
        for ent in doc2.ents:
            print(ent.label_, ent.text)




## === cell 1
spacy.prefer_gpu()

n_iter = 20

modelsdir = "models"
datadir = "/kaggle/input/tweet-sentiment-extraction"

train, test, _ = read_data(datadir)
positive, negative = training_data(train)




## === cell 2
os.makedirs(modelsdir, exist_ok=True)

positive_model_path = os.path.join(modelsdir, "positive")
negative_model_path = os.path.join(modelsdir, "negative")

print("Training positive model...")
train_model(positive, "positive", output_dir=positive_model_path, n_iter=n_iter)

print("\nTraining negative model...")
train_model(negative, "negative", output_dir=negative_model_path, n_iter=n_iter)




## === cell 3
run_model(test, positive_model_path, negative_model_path, outputpath="submission.csv")
