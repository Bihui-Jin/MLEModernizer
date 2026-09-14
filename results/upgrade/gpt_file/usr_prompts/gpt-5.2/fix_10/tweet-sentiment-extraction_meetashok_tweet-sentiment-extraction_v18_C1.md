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
    data = dataframe.dropna(subset=["text", "selected_text", "sentiment"]).copy()
    data["text"] = data["text"].astype(str).str.lower()
    data["selected_text"] = data["selected_text"].astype(str).str.lower()

    texts = data["text"].to_numpy()
    sels = data["selected_text"].to_numpy()
    starts = [t.find(s) for t, s in zip(texts, sels)]
    data["start"] = starts
    data["end"] = data["start"] + data["selected_text"].str.len()

    valid = (data["start"] >= 0) & (data["end"] > data["start"])
    data = data.loc[valid, ["text", "sentiment", "start", "end"]]

    pos_df = data[data["sentiment"] == "positive"]
    neg_df = data[data["sentiment"] == "negative"]

    positive = list(
        zip(
            pos_df["text"].tolist(),
            [
                {"entities": [(int(s), int(e), LABEL)]}
                for s, e in zip(pos_df["start"].to_numpy(), pos_df["end"].to_numpy())
            ],
        )
    )
    negative = list(
        zip(
            neg_df["text"].tolist(),
            [
                {"entities": [(int(s), int(e), LABEL)]}
                for s, e in zip(neg_df["start"].to_numpy(), neg_df["end"].to_numpy())
            ],
        )
    )

    print(f"Positive data size: {len(positive):,}")
    print(f"Negative data size: {len(negative):,}")

    return positive, negative


def run_model(data, positivemodel, negativemodel, outputpath=None):
    print("Loading models...")
    positivemodel = str(Path(positivemodel).resolve())
    negativemodel = str(Path(negativemodel).resolve())
    positive_nlp = spacy.load(positivemodel)
    negative_nlp = spacy.load(negativemodel)

    data = data.dropna(subset=["text", "sentiment"]).reset_index(drop=True)

    input_train = "selected_text" in data.columns

    text_series = data["text"].astype(str)
    sent_series = data["sentiment"].astype(str)

    short_mask = text_series.str.split().str.len().le(2).to_numpy()
    sent = sent_series.to_numpy()
    texts = text_series.to_numpy()

    selected_texts = [""] * len(data)

    neutral_idx = (sent == "neutral").nonzero()[0]
    for i in neutral_idx:
        selected_texts[i] = texts[i]

    pos_short_idx = ((sent == "positive") & short_mask).nonzero()[0]
    for i in pos_short_idx:
        selected_texts[i] = texts[i]
    neg_short_idx = ((sent == "negative") & short_mask).nonzero()[0]
    for i in neg_short_idx:
        selected_texts[i] = texts[i]

    pos_long_idx = ((sent == "positive") & (~short_mask)).nonzero()[0]
    if pos_long_idx.size > 0:
        pos_text_gen = (texts[i] for i in pos_long_idx.tolist())
        for i, doc in zip(
            pos_long_idx.tolist(), positive_nlp.pipe(pos_text_gen, batch_size=128)
        ):
            selected_texts[i] = doc.ents[0].text if len(doc.ents) > 0 else texts[i]

    neg_long_idx = ((sent == "negative") & (~short_mask)).nonzero()[0]
    if neg_long_idx.size > 0:
        neg_text_gen = (texts[i] for i in neg_long_idx.tolist())
        for i, doc in zip(
            neg_long_idx.tolist(), negative_nlp.pipe(neg_text_gen, batch_size=128)
        ):
            selected_texts[i] = doc.ents[0].text if len(doc.ents) > 0 else texts[i]

    output_df = pd.DataFrame(
        {"textID": data["textID"].tolist(), "selected_text": selected_texts}
    )

    if not input_train:
        if not outputpath:
            suffix = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            os.makedirs("submissions", exist_ok=True)
            outputpath = os.path.join("submissions", "submission_" + suffix + ".csv")

        print(f"Saving submission to {outputpath} ...")
        output_df.to_csv(outputpath, index=False)

    if input_train:
        jaccards = {"positive": [], "negative": [], "neutral": []}
        for txt, sentv, sel in zip(
            text_series.tolist(), sent_series.tolist(), selected_texts
        ):
            jaccards[sentv].append(jaccard_similarity(txt, sel))

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
    spacy.util.fix_random_seed(0)

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

    init_tuples = traindata[: min(200, len(traindata))]

    def init_examples():
        examples = []
        for text, ann in init_tuples:
            doc = nlp.make_doc(text)
            examples.append(Example.from_dict(doc, ann))
        return examples

    optimizer = nlp.initialize(get_examples=init_examples)

    with nlp.disable_pipes(*other_pipes):
        sizes = compounding(1.0, 4.0, 1.001)
        for itn in range(n_iter):
            random.shuffle(traindata)
            losses = {}

            for batch in minibatch(traindata, size=sizes):
                texts = [text for text, _ in batch]
                annotations = [ann for _, ann in batch]
                docs = list(nlp.pipe(texts, batch_size=len(texts)))
                examples = [
                    Example.from_dict(doc, ann) for doc, ann in zip(docs, annotations)
                ]
                nlp.update(examples, sgd=optimizer, drop=0.35, losses=losses)

            print("Losses", losses)

    test_text = "i`d have responded, if i were going"
    doc = nlp(test_text)
    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)

    if output_dir is not None:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        print("Loading from", output_dir)
        nlp2 = spacy.load(str(output_dir))
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

sub = pd.read_csv("submission.csv")
print("Wrote submission.csv; head:")
print(sub.head())
print("submission.csv shape:", sub.shape)
print("submission.csv columns:", list(sub.columns))
assert list(sub.columns) == ["textID", "selected_text"]
assert sub.shape[0] == test.shape[0]
assert str(Path("submission.csv")).endswith(".csv")
