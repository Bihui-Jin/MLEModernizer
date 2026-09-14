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

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
sklearn-pandas==2.2.0
tqdm==4.67.1

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

0.18989

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import json
import os
import difflib

import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text as sklearn_text
from scipy import spatial


def _levenshtein_ratio(s1: str, s2: str) -> float:
    """Return a similarity ratio between 0 and 1 using difflib."""
    return difflib.SequenceMatcher(None, s1, s2).ratio()


try:
    from Levenshtein import ratio as levenshtein_distance
except Exception:  # pragma: no cover
    levenshtein_distance = _levenshtein_ratio

r_buf = [
    "is",
    "are",
    "do",
    "does",
    "did",
    "was",
    "were",
    "will",
    "can",
    "the",
    "a",
    "of",
    "in",
    "and",
    "on",
    "what",
    "where",
    "when",
    "which",
]


def clean(x: str) -> str:
    x = x.lower()
    for r in r_buf:
        x = x.replace(r, "")
    return x


bin_question_tokens = ["is", "are", "do", "does", "did", "was", "were", "will", "can"]
stop_words = sklearn_text.ENGLISH_STOP_WORDS.union(["book"])


def predict(json_data, annotated=False):
    """Return long‑answer span, short answer (YES/NO/blank) and related texts."""
    candidates = [c for c in json_data["long_answer_candidates"] if c["top_level"]]
    doc_tokenized = json_data["document_text"].split(" ")
    question = json_data["question_text"]
    question_tokens = question.split(" ")
    if annotated:
        ann = json_data["annotations"][0]

    tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words=stop_words)
    tfidf.fit([json_data["document_text"]])
    q_tfidf = tfidf.transform([question]).todense()

    scores = []
    for c in candidates:
        s, e = c["start_token"], c["end_token"]
        t = " ".join(doc_tokenized[s:e])
        t_tfidf = tfidf.transform([t]).todense()
        score = 1 - spatial.distance.cosine(q_tfidf, t_tfidf)
        scores.append(score)

    best_idx = int(np.argmax(scores))
    ans = candidates[best_idx]
    ans_long = f"{ans['start_token']}:{ans['end_token']}"

    if question_tokens[0] in bin_question_tokens:
        ans_short = "YES"
    else:
        ans_short = ""

    if annotated:
        ann_long_text = " ".join(
            doc_tokenized[
                ann["long_answer"]["start_token"] : ann["long_answer"]["end_token"]
            ]
        )
        if ann["yes_no_answer"] == "NONE":
            if ann["short_answers"]:
                sa = ann["short_answers"][0]
                ann_short_text = " ".join(
                    doc_tokenized[sa["start_token"] : sa["end_token"]]
                )
            else:
                ann_short_text = ""
        else:
            ann_short_text = ann["yes_no_answer"]
    else:
        ann_long_text = ""
        ann_short_text = ""

    ans_long_text = " ".join(doc_tokenized[ans["start_token"] : ans["end_token"]])
    ans_short_text = ans_short if ans_short else ""

    return (
        ans_long,
        ans_short,
        question,
        ann_long_text,
        ann_short_text,
        ans_long_text,
        ans_short_text,
    )




## === cell 1
ids = []
anns = []
preds = []

questions = []
ann_texts = []
ans_texts = []

n_samples = 500  # limited for quick run

train_path = "/kaggle/input/tensorflow2-question-answering/simplified-nq-train.jsonl"
with open(train_path, "r") as json_file:
    for cnt, line in enumerate(tqdm(json_file, total=n_samples)):
        json_data = json.loads(line)

        ids.append(f"{json_data['example_id']}_long")
        ids.append(f"{json_data['example_id']}_short")

        l_ans = (
            f"{json_data['annotations'][0]['long_answer']['start_token']}:"
            f"{json_data['annotations'][0]['long_answer']['end_token']}"
        )
        if json_data["annotations"][0]["yes_no_answer"] == "NONE":
            if json_data["annotations"][0]["short_answers"]:
                s = json_data["annotations"][0]["short_answers"][0]
                s_ans = f"{s['start_token']}:{s['end_token']}"
            else:
                s_ans = ""
        else:
            s_ans = json_data["annotations"][0]["yes_no_answer"]

        anns.append(l_ans)
        anns.append(s_ans)

        (
            l_pred,
            s_pred,
            question,
            ann_long_text,
            ann_short_text,
            ans_long_text,
            ans_short_text,
        ) = predict(json_data, annotated=True)

        preds.append(l_pred)
        preds.append(s_pred)

        questions.append(question)
        questions.append(question)

        ann_texts.append(ann_long_text)
        ann_texts.append(ann_short_text)

        ans_texts.append(ans_long_text)
        ans_texts.append(ans_short_text)

        if cnt + 1 >= n_samples:
            break

train_ann = pd.DataFrame(
    {
        "example_id": ids,
        "question": questions,
        "CorrectString": anns,
        "CorrectText": ann_texts,
        "PredictionString": preds,
        "PredictionText": ans_texts,
    }
)
train_ann.to_csv("train_data.csv", index=False)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
InvalidParameterError                     Traceback (most recent call last)
/tmp/ipykernel_11/3800879461.py in <cell line: 0>()
     44             ans_long_text,
     45             ans_short_text,
---> 46         ) = predict(json_data, annotated=True)
     47 
     48         preds.append(l_pred)

/tmp/ipykernel_11/2004608667.py in predict(json_data, annotated)
     74     # TF‑IDF on the whole document (single document)
     75     tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words=stop_words)
---> 76     tfidf.fit([json_data["document_text"]])
     77     q_tfidf = tfidf.transform([question]).todense()
     78 

/usr/local/lib/python3.11/dist-packages/sklearn/feature_extraction/text.py in fit(self, raw_documents, y)
   2092             Fitted vectorizer.
   2093         """
-> 2094         self._validate_params()
   2095         self._check_params()
   2096         self._warn_for_unused_params()

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_params(self)
    598         accepted constraints.
    599         """
--> 600         validate_parameter_constraints(
    601             self._parameter_constraints,
    602             self.get_params(deep=False),

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_param_validation.py in validate_parameter_constraints(parameter_constraints, params, caller_name)
     95                 )
     96 
---> 97             raise InvalidParameterError(
     98                 f"The {param_name!r} parameter of {caller_name} must be"
     99                 f" {constraints_str}. Got {param_val!r} instead."

InvalidParameterError: The 'stop_words' parameter of TfidfVectorizer must be a str among {'english'}, an instance of 'list' or None. Got frozenset({'above', 'upon', 'nobody', 'latterly', 'will', 'he', 'whatever', 'seem', 'former', 'might', 'every', 'these', 'am', 'five', 'against', 'something', 'see', 'because', 'back', 'at', 'whereupon', 'how', 'next', 'least', 'whether', 'twenty', 'whole', 'without', 'de', 'do', 'either', 'thru', 'anywhere', 'formerly', 'enough', 'perhaps', 'nothing', 'have', 'your', 'ten', 'via', 'cry', 'them', 'can', 'must', 'six', 'the', 'latter', 'whenever', 'bill', 'himself', 'un', 'therefore', 'those', 'then', 'whereas', 'anyhow', 'seems', 'cant', 'con', 'ourselves', 'take', 'we', 'should', 'within', 'forty', 'once', 'any', 'somewhere', 'therein', 'mine', 'describe', 'moreover', 'anyone', 'detail', 'somehow', 'few', 'also', 'several', 'very', 'done', 'already', 'through', 'neither', 're', 'others', 'thereupon', 'again', 'fifty', 'sincere', 'amoungst', 'it', 'for', 'rather', 'another', 'fire', 'about', 'us', 'indeed', 'over', 'find', 'almost', 'became', 'together', 'name', 'nowhere', 'book', 'than', 'everyone', 'twelve', 'now', 'often', 'wherein', 'behind', 'become', 'thick', 'part', 'most', 'but', 'there', 'this', 'amongst', 'off', 'all', 'same', 'with', 'since', 'were', 'seeming', 'whence', 'thin', 'its', 'though', 'becoming', 'ever', 'seemed', 'more', 'being', 'too', 'not', 'may', 'thereby', 'myself', 'hereafter', 'per', 'from', 'while', 'less', 'system', 'themselves', 'call', 'empty', 'here', 'mill', 'a', 'inc', 'their', 'due', 'during', 'me', 'whither', 'as', 'top', 'around', 'had', 'full', 'an', 'by', 'still', 'are', 'sometimes', 'only', 'whom', 'has', 'otherwise', 'everywhere', 'after', 'yours', 'nine', 'never', 'she', 'before', 'bottom', 'whereby', 'what', 'three', 'sixty', 'much', 'fill', 'fifteen', 'everything', 'would', 'else', 'nor', 'alone', 'up', 'beforehand', 'always', 'none', 'into', 'two', 'yourself', 'anything', 'out', 'elsewhere', 'could', 'no', 'found', 'ie', 'they', 'yet', 'except', 'etc', 'own', 'under', 'made', 'of', 'eg', 'to', 'in', 'when', 'that', 'both', 'is', 'nevertheless', 'toward', 'was', 'third', 'noone', 'you', 'hereby', 'wherever', 'beyond', 'his', 'afterwards', 'between', 'why', 'someone', 'side', 'last', 'along', 'yourselves', 'four', 'put', 'throughout', 'cannot', 'please', 'eight', 'on', 'thus', 'show', 'move', 'who', 'itself', 'which', 'whose', 'him', 'be', 'many', 'hence', 'give', 'each', 'some', 'becomes', 'if', 'hers', 'ltd', 'my', 'go', 'anyway', 'beside', 'mostly', 'interest', 'other', 'meanwhile', 'been', 'where', 'towards', 'whoever', 'front', 'across', 'one', 'whereafter', 'sometime', 'amount', 'first', 'keep', 'even', 'further', 'although', 'herself', 'co', 'eleven', 'and', 'so', 'get', 'such', 'onto', 'herein', 'couldnt', 'i', 'among', 'hereupon', 'our', 'thence', 'ours', 'namely', 'her', 'serious', 'down', 'below', 'hundred', 'hasnt', 'or', 'thereafter', 'until', 'besides', 'well', 'however'}) instead.

## === cell 2
f1 = f1_score(
    train_ann["CorrectString"].values,
    train_ann["PredictionString"].values,
    average="micro",
)
print(f"F1-score (sampled): {f1:.4f}")



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/939246339.py in <cell line: 0>()
      1 # Evaluate micro F1 on the sampled validation slice
      2 f1 = f1_score(
----> 3     train_ann["CorrectString"].values,
      4     train_ann["PredictionString"].values,
      5     average="micro",

NameError: name 'train_ann' is not defined

## === cell 3
ids = []
preds = []
questions = []
ans_texts = []

test_path = "/kaggle/input/tensorflow2-question-answering/simplified-nq-test.jsonl"
with open(test_path, "r") as json_file:
    for line in tqdm(json_file):
        json_data = json.loads(line)

        ids.append(f"{json_data['example_id']}_long")
        ids.append(f"{json_data['example_id']}_short")

        l_pred, s_pred, question, _, _, ans_long_text, ans_short_text = predict(
            json_data, annotated=False
        )

        preds.append(l_pred)
        preds.append(s_pred)

        questions.append(question)
        questions.append(question)

        ans_texts.append(ans_long_text)
        ans_texts.append(ans_short_text)

subm = pd.DataFrame(
    {
        "example_id": ids,
        "PredictionString": preds,
        "PredictionText": ans_texts,
        "question": questions,
    }
)
subm.to_csv("test_data.csv", index=False)
subm[["example_id", "PredictionString"]].to_csv("submission.csv", index=False)
print("Submission written to submission.csv")

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
InvalidParameterError                     Traceback (most recent call last)
/tmp/ipykernel_11/1308755207.py in <cell line: 0>()
     13         ids.append(f"{json_data['example_id']}_short")
     14 
---> 15         l_pred, s_pred, question, _, _, ans_long_text, ans_short_text = predict(
     16             json_data, annotated=False
     17         )

/tmp/ipykernel_11/2004608667.py in predict(json_data, annotated)
     74     # TF‑IDF on the whole document (single document)
     75     tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words=stop_words)
---> 76     tfidf.fit([json_data["document_text"]])
     77     q_tfidf = tfidf.transform([question]).todense()
     78 

/usr/local/lib/python3.11/dist-packages/sklearn/feature_extraction/text.py in fit(self, raw_documents, y)
   2092             Fitted vectorizer.
   2093         """
-> 2094         self._validate_params()
   2095         self._check_params()
   2096         self._warn_for_unused_params()

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_params(self)
    598         accepted constraints.
    599         """
--> 600         validate_parameter_constraints(
    601             self._parameter_constraints,
    602             self.get_params(deep=False),

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_param_validation.py in validate_parameter_constraints(parameter_constraints, params, caller_name)
     95                 )
     96 
---> 97             raise InvalidParameterError(
     98                 f"The {param_name!r} parameter of {caller_name} must be"
     99                 f" {constraints_str}. Got {param_val!r} instead."

InvalidParameterError: The 'stop_words' parameter of TfidfVectorizer must be a str among {'english'}, an instance of 'list' or None. Got frozenset({'above', 'upon', 'nobody', 'latterly', 'will', 'he', 'whatever', 'seem', 'former', 'might', 'every', 'these', 'am', 'five', 'against', 'something', 'see', 'because', 'back', 'at', 'whereupon', 'how', 'next', 'least', 'whether', 'twenty', 'whole', 'without', 'de', 'do', 'either', 'thru', 'anywhere', 'formerly', 'enough', 'perhaps', 'nothing', 'have', 'your', 'ten', 'via', 'cry', 'them', 'can', 'must', 'six', 'the', 'latter', 'whenever', 'bill', 'himself', 'un', 'therefore', 'those', 'then', 'whereas', 'anyhow', 'seems', 'cant', 'con', 'ourselves', 'take', 'we', 'should', 'within', 'forty', 'once', 'any', 'somewhere', 'therein', 'mine', 'describe', 'moreover', 'anyone', 'detail', 'somehow', 'few', 'also', 'several', 'very', 'done', 'already', 'through', 'neither', 're', 'others', 'thereupon', 'again', 'fifty', 'sincere', 'amoungst', 'it', 'for', 'rather', 'another', 'fire', 'about', 'us', 'indeed', 'over', 'find', 'almost', 'became', 'together', 'name', 'nowhere', 'book', 'than', 'everyone', 'twelve', 'now', 'often', 'wherein', 'behind', 'become', 'thick', 'part', 'most', 'but', 'there', 'this', 'amongst', 'off', 'all', 'same', 'with', 'since', 'were', 'seeming', 'whence', 'thin', 'its', 'though', 'becoming', 'ever', 'seemed', 'more', 'being', 'too', 'not', 'may', 'thereby', 'myself', 'hereafter', 'per', 'from', 'while', 'less', 'system', 'themselves', 'call', 'empty', 'here', 'mill', 'a', 'inc', 'their', 'due', 'during', 'me', 'whither', 'as', 'top', 'around', 'had', 'full', 'an', 'by', 'still', 'are', 'sometimes', 'only', 'whom', 'has', 'otherwise', 'everywhere', 'after', 'yours', 'nine', 'never', 'she', 'before', 'bottom', 'whereby', 'what', 'three', 'sixty', 'much', 'fill', 'fifteen', 'everything', 'would', 'else', 'nor', 'alone', 'up', 'beforehand', 'always', 'none', 'into', 'two', 'yourself', 'anything', 'out', 'elsewhere', 'could', 'no', 'found', 'ie', 'they', 'yet', 'except', 'etc', 'own', 'under', 'made', 'of', 'eg', 'to', 'in', 'when', 'that', 'both', 'is', 'nevertheless', 'toward', 'was', 'third', 'noone', 'you', 'hereby', 'wherever', 'beyond', 'his', 'afterwards', 'between', 'why', 'someone', 'side', 'last', 'along', 'yourselves', 'four', 'put', 'throughout', 'cannot', 'please', 'eight', 'on', 'thus', 'show', 'move', 'who', 'itself', 'which', 'whose', 'him', 'be', 'many', 'hence', 'give', 'each', 'some', 'becomes', 'if', 'hers', 'ltd', 'my', 'go', 'anyway', 'beside', 'mostly', 'interest', 'other', 'meanwhile', 'been', 'where', 'towards', 'whoever', 'front', 'across', 'one', 'whereafter', 'sometime', 'amount', 'first', 'keep', 'even', 'further', 'although', 'herself', 'co', 'eleven', 'and', 'so', 'get', 'such', 'onto', 'herein', 'couldnt', 'i', 'among', 'hereupon', 'our', 'thence', 'ours', 'namely', 'her', 'serious', 'down', 'below', 'hundred', 'hasnt', 'or', 'thereafter', 'until', 'besides', 'well', 'however'}) instead.
