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
For each article + question pair, you must predict / select long and short form answers to the question drawn *directly from the article*.

- A long answer would be a longer section of text that answers the question - several sentences or a paragraph.
- A short answer might be a sentence or phrase, or even in some cases a YES/NO. The short answers are always contained within / a subset of one of the plausible long answers.
- A given article can (and very often will) allow for both long *and* short answers, depending on the question.

There is more detail about the data and what you're predicting [on the Github page for the Natural Questions dataset](https://github.com/google-research-datasets/natural-questions/blob/master/README.md). This page also contains helpful utilities and scripts. Note that we are using the simplified text version of the data - most of the HTML tags have been removed, and only those necessary to break up paragraphs / sections are included.

## Metric
Micro F1. Predicted long and short answers must match exactly the token indices of one of the ground truth labels ((or match YES/NO if the question has a yes/no short answer). There may be up to five labels for long answers, and more for short. If no answer applies, leave the prediction blank/null.

## Submission Format
For each ID in the test set, you must predict a) a set of start:end token indices, b) a YES/NO answer if applicable (short answers ONLY), or c) a BLANK answer if no prediction can be made. The file should contain a header and have the following format:

```
-7853356005143141653_long,6:18
-7853356005143141653_short,YES
-545833482873225036_long,105:200
-545833482873225036_short,
-6998273848279890840_long,
-6998273848279890840_short,NO
```
`
## Data
Each sample contains a Wikipedia article, a related question, and the candidate long form answers. The training examples also provide the correct long and short form answer or answers for the sample, if any exist.

- **simplified-nq-train.jsonl** - the training data, in newline-delimited JSON format.
- **simplified-nq-kaggle-test.jsonl** - the test data, in newline-delimited JSON format.
- **sample_submission.csv** - a sample submission file in the correct format

### Data fields
- **document_text** - the text of the article in question (with some HTML tags to provide document structure). The text can be tokenized by splitting on whitespace.
- **question_text** - the question to be answered
- **long_answer_candidates** - a JSON array containing all of the plausible long answers.
- **annotations** - a JSON array containing all of the correct long + short answers. Only provided for train.
- **document_url** - the URL for the full article. Provided for informational purposes only. This is NOT the simplified version of the article so indices from this cannot be used directly. The content may also no longer match the html used to generate document_text. Only provided for train.
- **example_id** - unique ID for the sample.

# 2. Python version

3.8

# 3. Installed packages

beautifulsoup4==4.13.4
geopandas==0.14.4
nltk==3.9.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (173 lines)
            sample_submission.csv (61477 lines)
            sample_submission.csv.zip (460.6 kB)
            simplified-nq-test.jsonl (1.7 GB)
            simplified-nq-train.jsonl (15.7 GB)
            tensorflow2-question-answering/
                description.md (173 lines)
                sample_submission.csv (61477 lines)
                ... and 3 other files
                tensorflow2-question-answering/
        input/
            description.md (173 lines)
            sample_submission.csv (61477 lines)
            sample_submission.csv.zip (460.6 kB)
            simplified-nq-test.jsonl (1.7 GB)
            simplified-nq-train.jsonl (15.7 GB)
            tensorflow2-question-answering/
                description.md (173 lines)
                sample_submission.csv (61477 lines)
                ... and 3 other files
                tensorflow2-question-answering/
        working/
            tensorflow2-question-answering/
                description.md (173 lines)
                sample_submission.csv (61477 lines)
                ... and 3 other files
                tensorflow2-question-answering/
```

-> data/sample_submission.csv has 61476 rows and 2 columns.
Here is some information about the columns:
PredictionString (float64) has 0 unique values: [nan]
example_id (object) has 2000 unique values. Some example values: ['-4639148749459150090_long', '-257485578111885185_short', '-257485578111885185_long', '-9110190923673509457_short']

-> data/tensorflow2-question-answering/sample_submission.csv has 61476 rows and 2 columns.
Here is some information about the columns:
PredictionString (float64) has 0 unique values: [nan]
example_id (object) has 2000 unique values. Some example values: ['-4639148749459150090_long', '-257485578111885185_short', '-257485578111885185_long', '-9110190923673509457_short']

-> input/sample_submission.csv has 61476 rows and 2 columns.
Here is some information about the columns:
PredictionString (float64) has 0 unique values: [nan]
example_id (object) has 2000 unique values. Some example values: ['-4639148749459150090_long', '-257485578111885185_short', '-257485578111885185_long', '-9110190923673509457_short']

-> input/tensorflow2-question-answering/sample_submission.csv has 61476 rows and 2 columns.
Here is some information about the columns:
PredictionString (float64) has 0 unique values: [nan]
example_id (object) has 2000 unique values. Some example values: ['-4639148749459150090_long', '-257485578111885185_short', '-257485578111885185_long', '-9110190923673509457_short']

-> working/tensorflow2-question-answering/sample_submission.csv has 61476 rows and 2 columns.
Here is some information about the columns:
PredictionString (float64) has 0 unique values: [nan]
example_id (object) has 2000 unique values. Some example values: ['-4639148749459150090_long', '-257485578111885185_short', '-257485578111885185_long', '-9110190923673509457_short']

# 5. Target score

0.09758

# 6. Current score

0.03637

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.03637) has done: 'The timeout is dominated by parsing 1.7GB JSONL into a full DataFrame and then running BeautifulSoup HTML parsing for every test row. I make the pipeline streaming: read only `example_id`, `question_text`, and `document_text` line-by-line from the JSONL, generate predictions on the fly, and never materialize the full test DataFrame. I also remove BeautifulSoup usage and extract paragraph text with a compiled regex, which is equivalent for this simplified HTML and preserves the same “pick a paragraph then find it in the original doc” logic. Finally, I speed up word matching by using set intersections rather than nested Python loops, while keeping the same scoring semantics.'

# 9. Code solution

## === cell 0
import os, json, random, re
import numpy as np
import pandas as pd

print("Python OK. Not importing TensorFlow to avoid protobuf MessageFactory crash.")

random.seed(12345)
np.random.seed(12345)




## === cell 1
def read_jsonl(path, max_limit=None):
    """Read jsonl into DataFrame. If max_limit is None, read full file."""
    rows = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            rows.append(json.loads(line))
            if max_limit is not None and (i + 1) >= max_limit:
                break
    return pd.DataFrame(rows)


p = "../input/tensorflow2-question-answering/"

train = read_jsonl(p + "simplified-nq-train.jsonl", max_limit=4000)
sub = pd.read_csv(p + "sample_submission.csv")

train["D"] = [t[0]["long_answer"]["start_token"] for t in train.annotations]
train = train[train["D"] > -1].reset_index(drop=True)

print("train/sub shapes:", train.shape, sub.shape)




## === cell 2
i = min(99, len(train) - 1)
print("URL:", train.document_url[i])
print(train.question_text[i])
print(train.long_answer_candidates[i][0])
doc_tokens = train.document_text[i].split()
la_start = train.annotations[i][0]["long_answer"]["start_token"]
la_end = train.annotations[i][0]["long_answer"]["end_token"]
print("LONG:", " ".join(doc_tokens[la_start:la_end])[:300], "...")
if len(train.annotations[i][0]["short_answers"]) > 0:
    sa0 = train.annotations[i][0]["short_answers"][0]
    print("SHORT:", " ".join(doc_tokens[sa0["start_token"] : sa0["end_token"]]))




## === cell 3
la = [
    t[0]["long_answer"]["end_token"] - t[0]["long_answer"]["start_token"]
    for t in train.annotations
]
sa = [
    t[0]["short_answers"][0]["end_token"] - t[0]["short_answers"][0]["start_token"]
    for t in train.annotations
    if len(t[0]["short_answers"]) > 0
]
print("Median long/short lengths:", np.median(la), np.median(sa))




## === cell 4
import nltk
from nltk.corpus import stopwords


def _get_stopwords():
    try:
        sw = set(stopwords.words("english"))
        if len(sw) > 0:
            return sw
    except Exception:
        pass
    try:
        nltk.download("stopwords", quiet=True)
        sw = set(stopwords.words("english"))
        if len(sw) > 0:
            return sw
    except Exception:
        pass
    return {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "if",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "as",
        "by",
        "at",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "it",
        "this",
        "that",
        "these",
        "those",
        "he",
        "she",
        "they",
        "we",
        "you",
        "i",
        "me",
        "my",
        "your",
        "our",
        "their",
        "his",
        "her",
        "its",
        "not",
        "no",
        "yes",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "will",
        "would",
        "can",
        "could",
        "should",
        "may",
        "might",
        "about",
        "into",
        "over",
        "under",
    }


STOPWORDS = _get_stopwords()

_P_TAG_RE = re.compile(r"<p[^>]*>(.*?)</p>", flags=re.IGNORECASE | re.DOTALL)
_WS_RE = re.compile(r"\s+")


def extract_paragraphs(doc_html, min_len=50):
    paras = []
    for m in _P_TAG_RE.finditer(doc_html):
        txt = m.group(1)
        txt = re.sub(r"<[^>]+>", " ", txt)
        txt = _WS_RE.sub(" ", txt).strip()
        if len(txt) > min_len:
            paras.append(txt)
    return paras


def qa_word_match(q, a_list):
    q_words = [w for w in q.lower().split() if w not in STOPWORDS]
    q_set = set(q_words)
    tm = 0
    best = a_list[0] if len(a_list) else ""
    for a in a_list:
        m = 0
        for w in a.lower().split():
            if w in q_set:
                m += 1
        if m > tm:
            tm = int(m)
            best = str(a)
    return best


aux_q_words = set(
    [
        "am",
        "are",
        "can",
        "could",
        "did",
        "do",
        "does",
        "has",
        "have",
        "is",
        "may",
        "should",
        "was",
        "were",
        "will",
    ]
)




## === cell 5
test_path = p + "simplified-nq-test.jsonl"

result_map = {}

with open(test_path, "r") as f:
    for line in f:
        ex = json.loads(line)
        base_id = str(ex["example_id"])
        doc = ex["document_text"]
        q = ex["question_text"]

        paragraphs = extract_paragraphs(doc, min_len=50)

        if len(paragraphs) > 0:
            chosen = qa_word_match(q, paragraphs)
            pos = doc.find(chosen)
            if pos < 0:
                r = 0
            else:
                r = len(doc[:pos].split()) - 1
                if r < 0:
                    r = 0
            long_answer = f"{r}:{r + len(paragraphs[0].split()) + 2}"
        else:
            doc_len = len(doc.split())
            try:
                r = random.randrange(390, doc_len)
            except Exception:
                r = 7
            long_answer = f"{r}:{r + 114}"

        q_tokens = set(q.lower().split())
        if len(aux_q_words.intersection(q_tokens)) > 0:
            short_answer = random.choice(["YES", "NO"])
        else:
            try:
                rr = random.randrange(r, r + 114)
            except Exception:
                rr = r
            short_answer = f"{rr}:{rr + 2}"

        result_map[base_id + "_long"] = long_answer
        result_map[base_id + "_short"] = short_answer




## === cell 6
submission = sub.copy()
submission["PredictionString"] = submission["example_id"].map(result_map)
submission["PredictionString"] = submission["PredictionString"].fillna("")

out_path = "submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print("submission shape:", submission.shape)
print(submission.head())
print("Non-empty predictions:", (submission["PredictionString"] != "").sum())
