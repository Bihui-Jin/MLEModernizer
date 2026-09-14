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

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import datetime
import pandas as pd
import spacy
from tqdm import tqdm


def read_data(datadir):
    """Read train and test CSVs from the provided directory."""
    train_path = os.path.join(datadir, "train.csv")
    test_path = os.path.join(datadir, "test.csv")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test, None


def training_data(train):
    """Split the training set into positive and negative subsets."""
    positive = train[train.sentiment == "positive"].reset_index(drop=True)
    negative = train[train.sentiment == "negative"].reset_index(drop=True)
    return positive, negative


def train_model(data, sentiment, model=None, output_dir=None, n_iter=30):
    """
    Placeholder for the original training routine.
    It creates an empty directory so that `spacy.load` can later detect the
    absence of a model and fall back to the simple heuristic.
    """
    os.makedirs(output_dir, exist_ok=True)
    return None


def _recover_original_case(original_text, snippet):
    """Return the snippet unchanged – the original code only needed case fixing."""
    return snippet


def jaccard_similarity(text, pred):
    """Token‑set Jaccard similarity between the whole tweet and the predicted snippet."""
    set_a = set(text.lower().split())
    set_b = set(pred.lower().split())
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def _is_too_short(text, min_words=2):
    """Return True if the predicted snippet has fewer than min_words tokens."""
    return len(text.split()) < min_words


_POSITIVE_KEYWORDS = {
    "good",
    "great",
    "nice",
    "love",
    "awesome",
    "fantastic",
    "excellent",
    "wonderful",
}
_NEGATIVE_KEYWORDS = {
    "bad",
    "worst",
    "terrible",
    "hate",
    "awful",
    "poor",
    "sad",
    "angry",
}


def _fallback_snippet(row):
    """Pick a short phrase (up to three words) containing a sentiment keyword if possible."""
    tokens = row.text.split()
    lowered = [t.lower().strip(".,!?\"'") for t in tokens]

    def phrase_around(i):
        start = max(i - 1, 0)
        end = min(i + 2, len(tokens))  # exclusive
        return " ".join(tokens[start:end])

    if row.sentiment == "positive":
        for i, w in enumerate(lowered):
            if w in _POSITIVE_KEYWORDS:
                return phrase_around(i)
    elif row.sentiment == "negative":
        for i, w in enumerate(lowered):
            if w in _NEGATIVE_KEYWORDS:
                return phrase_around(i)
    return row.text




## === cell 1
spacy.prefer_gpu()
n_iter = 30
modelsdir = "models"
datadir = "/kaggle/input/tweet-sentiment-extraction"

train, test, _ = read_data(datadir)
positive, negative = training_data(train)




## === cell 2
os.makedirs(modelsdir, exist_ok=True)

positive_model_path = os.path.join(modelsdir, "positive")
negative_model_path = os.path.join(modelsdir, "negative")

print("Training positive model...")
train_model(
    positive, "positive", model=None, output_dir=positive_model_path, n_iter=n_iter
)

print("\nTraining negative model...")
train_model(
    negative, "negative", model=None, output_dir=negative_model_path, n_iter=n_iter
)




## === cell 3
def run_model(data, positivemodel, negativemodel, outputpath=None):
    print("Loading models...")
    if positivemodel is not None and os.path.isdir(positivemodel):
        try:
            positive_nlp = spacy.load(positivemodel)
        except Exception:
            positive_nlp = None
            print(
                f"Could not load positive model from '{positivemodel}'. Using fallback."
            )
    else:
        positive_nlp = None
        print(
            f"Positive model not found at '{positivemodel}'. Using fallback (heuristic)."
        )

    if negativemodel is not None and os.path.isdir(negativemodel):
        try:
            negative_nlp = spacy.load(negativemodel)
        except Exception:
            negative_nlp = None
            print(
                f"Could not load negative model from '{negativemodel}'. Using fallback."
            )
    else:
        negative_nlp = None
        print(
            f"Negative model not found at '{negativemodel}'. Using fallback (heuristic)."
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
                selected_text = _fallback_snippet(row)
            else:
                ents = positive_nlp(text_lc).ents
                selected_text = (
                    max(ents, key=lambda e: len(e.text)).text
                    if ents
                    else _fallback_snippet(row)
                )
        elif row.sentiment == "negative":
            if len(text_lc.split()) <= 2 or negative_nlp is None:
                selected_text = _fallback_snippet(row)
            else:
                ents = negative_nlp(text_lc).ents
                selected_text = (
                    max(ents, key=lambda e: len(e.text)).text
                    if ents
                    else _fallback_snippet(row)
                )
        else:
            selected_text = row.text

        if _is_too_short(selected_text):
            selected_text = row.text

        if selected_text != row.text:
            selected_text = _recover_original_case(row.text, selected_text)

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




## === cell 4
run_model(test, positive_model_path, negative_model_path, outputpath="submission.csv")

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1799682640.py in <cell line: 0>()
----> 1 run_model(test, positive_model_path, negative_model_path, outputpath="submission.csv")

/tmp/ipykernel_11/3376834202.py in run_model(data, positivemodel, negativemodel, outputpath)
     81             suffix = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
     82             outputpath = os.path.join("submissions", "submission_" + suffix + ".csv")
---> 83         os.makedirs(os.path.dirname(outputpath), exist_ok=True)
     84 
     85         print("Saving submission...")

/usr/lib/python3.11/os.py in makedirs(name, mode, exist_ok)

FileNotFoundError: [Errno 2] No such file or directory: ''
