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

# 5. Code solution

## === cell 0
from __future__ import unicode_literals, print_function

import os
import re
import string
import random
import datetime
from pathlib import Path

import pandas as pd
import spacy
from spacy.util import minibatch, compounding
from spacy.training.example import Example
from tqdm import tqdm

LABEL = "SELECTEDTEXT"


def read_data(datadir):
    train = pd.read_csv(os.path.join(datadir, "train.csv"))
    test = pd.read_csv(os.path.join(datadir, "test.csv"))
    sample_submission = pd.read_csv(os.path.join(datadir, "sample_submission.csv"))
    return (train, test, sample_submission)


def jaccard_similarity(string1, string2):
    wordset = lambda x: set(str(x).lower().split())
    a, b = wordset(string1), wordset(string2)
    if len(a) == 0 and len(b) == 0:
        return 1.0
    c = a.intersection(b)
    denom = len(a) + len(b) - len(c)
    return float(len(c)) / denom if denom != 0 else 0.0


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\n", "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    return text


def training_data(dataframe):
    data = (
        dataframe.dropna(subset=["text", "selected_text", "sentiment"])
        .assign(
            text=lambda x: x["text"].astype(str).str.lower(),
            selected_text=lambda x: x["selected_text"].astype(str).str.lower(),
        )
        .assign(
            start=lambda x: x.apply(lambda r: r.text.find(r.selected_text), axis=1),
        )
        .assign(
            end=lambda x: x.apply(lambda r: r.start + len(r.selected_text), axis=1),
        )
    )

    positive = []
    negative = []

    for _, row in data.iterrows():
        if row.end > row.start >= 0:
            train_row = (
                row.text,
                {"entities": [(int(row.start), int(row.end), LABEL)]},
            )
            if row.sentiment == "positive":
                positive.append(train_row)
            elif row.sentiment == "negative":
                negative.append(train_row)

    print(f"Positive data size: {len(positive):,}")
    print(f"Negative data size: {len(negative):,}")

    return positive, negative


def run_model(data, positivemodel, negativemodel, outputpath=None):
    print("Loading models...")
    positive_nlp = spacy.load(positivemodel)
    negative_nlp = spacy.load(negativemodel)

    data = data.dropna(subset=["text", "sentiment"])

    textIDs, selected_texts = [], []
    jaccards = {"positive": [], "negative": [], "neutral": []}

    input_train = "selected_text" in data.columns

    print("Iterating data...")
    for _, row in tqdm(data.iterrows(), total=len(data)):
        text = str(row.text)
        sent = row.sentiment

        if sent == "positive":
            if len(text.split()) <= 2:
                selected_text = text
            else:
                ents = positive_nlp(text).ents
                selected_text = ents[0].text if len(ents) > 0 else text
        elif sent == "negative":
            if len(text.split()) <= 2:
                selected_text = text
            else:
                ents = negative_nlp(text).ents
                selected_text = ents[0].text if len(ents) > 0 else text
        else:
            selected_text = text

        textIDs.append(row.textID)
        selected_texts.append(selected_text)

        if input_train:
            jaccard = jaccard_similarity(text, selected_text)
            jaccards[sent].append(jaccard)

    output_df = pd.DataFrame({"textID": textIDs, "selected_text": selected_texts})

    if not input_train:
        if not outputpath:
            suffix = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            os.makedirs("submissions", exist_ok=True)
            outputpath = os.path.join("submissions", "submission_" + suffix + ".csv")

        print(f"Saving submission to {outputpath} ...")
        output_df.to_csv(outputpath, index=False)

    if input_train:
        nums, dens = [], []
        for key in ("positive", "negative", "neutral"):
            num = sum(jaccards[key])
            den = len(jaccards[key]) if len(jaccards[key]) > 0 else 1
            print(f"Jaccard score for {key}: {num / den:.3f}")
            nums.append(num)
            dens.append(den)
        print(f"Jaccard score for overall: {sum(nums) / sum(dens):.3f}")


def train_model(traindata, new_model_name, model=None, output_dir=None, n_iter=30):
    """Train a spaCy NER model to extract selected text spans."""
    random.seed(0)

    if model is not None:
        nlp = spacy.load(model)
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")
        print("Created blank 'en' model")

    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    ner.add_label(LABEL)

    move_names = list(ner.move_names)
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    def make_examples(batch):
        examples = []
        for text, ann in batch:
            doc = nlp.make_doc(text)
            examples.append(Example.from_dict(doc, ann))
        return examples

    if model is None:
        init_examples = make_examples(traindata[: min(100, len(traindata))])
        optimizer = nlp.initialize(get_examples=lambda: init_examples)
    else:
        optimizer = nlp.resume_training()

    with nlp.disable_pipes(*other_pipes):
        sizes = compounding(1.0, 4.0, 1.001)
        for itn in range(n_iter):
            random.shuffle(traindata)
            batches = minibatch(traindata, size=sizes)
            losses = {}
            for batch in batches:
                examples = make_examples(batch)
                nlp.update(examples, sgd=optimizer, drop=0.35, losses=losses)
            print("Losses", losses)

    test_text = "i`d have responded, if i were going"
    doc = nlp(test_text)
    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        assert list(nlp2.get_pipe("ner").move_names) == move_names
        doc2 = nlp2(test_text)
        for ent in doc2.ents:
            print(ent.label_, ent.text)




## === cell 1
spacy.prefer_gpu()

n_iter = 20

datadir = "/kaggle/input/tweet-sentiment-extraction"
models_root = Path("models")
pos_dir = models_root / "positive"
neg_dir = models_root / "negative"

train, test, _ = read_data(datadir)
positive, negative = training_data(train)

models_root.mkdir(parents=True, exist_ok=True)

if not pos_dir.exists():
    print(f"Positive model not found at {pos_dir}. Training and saving...")
    train_model(
        positive,
        new_model_name="positive",
        model=None,
        output_dir=str(pos_dir),
        n_iter=n_iter,
    )

if not neg_dir.exists():
    print(f"Negative model not found at {neg_dir}. Training and saving...")
    train_model(
        negative,
        new_model_name="negative",
        model=None,
        output_dir=str(neg_dir),
        n_iter=n_iter,
    )



## === cell 2
run_model(
    test,
    str(pos_dir),
    str(neg_dir),
    outputpath="submission.csv",
)

print("Wrote submission.csv; head:")
print(pd.read_csv("submission.csv").head())
print("submission.csv shape:", pd.read_csv("submission.csv").shape)
print("submission.csv columns:", list(pd.read_csv("submission.csv").columns))
