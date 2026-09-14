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

0.27815

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import json
import numpy as np
import pandas as pd
import re
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from sklearn.metrics import f1_score

try:
    from tqdm import tqdm_notebook as tqdm
except ImportError:
    from tqdm import tqdm

try:
    from Levenshtein import ratio as levenshtein_distance
except ImportError:
    from difflib import SequenceMatcher

    def levenshtein_distance(a, b):
        """Return similarity ratio between two strings (0..1)."""
        return SequenceMatcher(None, a, b).ratio()


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from scipy import spatial




## === cell 1
n_answers = 1




## === cell 2
html_tags = ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""]
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
] + html_tags


def clean(x):
    """Lower‑case and remove stop tokens defined in r_buf."""
    x = x.lower()
    for r in r_buf:
        x = x.replace(r, "")
    x = re.sub(" +", " ", x)
    return x


bin_question_tokens = ["is", "are", "do", "does", "did", "was", "were", "will", "can"]
stop_words = text.ENGLISH_STOP_WORDS.union(["book"])


def predict(json_data, annotated=False):
    """Return long answer strings, short answer string and auxiliary info."""
    candidates = json_data["long_answer_candidates"]
    candidates = [c for c in candidates if c["top_level"] == True]
    doc_tokenized = json_data["document_text"].split(" ")
    question = json_data["question_text"]
    question_s = question.split(" ")
    if annotated:
        ann = json_data["annotations"][0]

    tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words=stop_words)
    tfidf.fit([json_data["document_text"]])
    q_tfidf = tfidf.transform([question]).todense()

    distances = []
    scores = []
    p_cnt = 1
    for i, c in enumerate(candidates):
        s, e = c["start_token"], c["end_token"]
        t = " ".join(doc_tokenized[s:e])
        distances.append(levenshtein_distance(clean(question), clean(t)))

        t_tfidf = tfidf.transform([t]).todense()
        score = 1 - spatial.distance.cosine(q_tfidf, t_tfidf)
        if np.isnan(score):
            score = 0.0

        if doc_tokenized[s] == "":
            score += 0.4**p_cnt
            p_cnt += 1
        scores.append(score)

    ans = (np.array(candidates)[np.argsort(scores)])[-n_answers:].tolist()

    if np.max(scores) < 0.2:
        ans_long = ["-1:-1"]
        ans = [{"start_token": 0, "end_token": 0}]
    else:
        ans_long = [str(a["start_token"]) + ":" + str(a["end_token"]) for a in ans]

    if question_s[0] in bin_question_tokens:
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
            if len(json_data["annotations"][0]["short_answers"]) > 0:
                ann_short_text = " ".join(
                    doc_tokenized[
                        ann["short_answers"][0]["start_token"] : ann["short_answers"][
                            0
                        ]["end_token"]
                    ]
                )
            else:
                ann_short_text = ""
        else:
            ann_short_text = ann["yes_no_answer"]
    else:
        ann_long_text = ""
        ann_short_text = ""

    ans_long_text = [
        " ".join(doc_tokenized[a["start_token"] : a["end_token"]]) for a in ans
    ]
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




## === cell 3
ids = []
anns = []
preds = []

questions = []
ann_texts = []
ans_texts = []

n_samples = 500

train_path = "/kaggle/input/tensorflow2-question-answering/simplified-nq-train.jsonl"
with open(train_path, "r") as json_file:
    cnt = 0
    for line in tqdm(json_file, total=n_samples):
        json_data = json.loads(line)

        l_ann = f"{json_data['annotations'][0]['long_answer']['start_token']}:{json_data['annotations'][0]['long_answer']['end_token']}"
        if json_data["annotations"][0]["yes_no_answer"] == "NONE":
            if len(json_data["annotations"][0]["short_answers"]) > 0:
                s_ann = f"{json_data['annotations'][0]['short_answers'][0]['start_token']}:{json_data['annotations'][0]['short_answers'][0]['end_token']}"
            else:
                s_ann = ""
        else:
            s_ann = json_data["annotations"][0]["yes_no_answer"]

        (
            l_ans,
            s_ans,
            question,
            ann_long_text,
            ann_short_text,
            ans_long_text,
            ans_short_text,
        ) = predict(json_data, annotated=True)

        ids += [f"{json_data['example_id']}_long"] * len(l_ans)
        ids.append(f"{json_data['example_id']}_short")

        anns += [l_ann] * len(l_ans)
        anns.append(s_ann)

        preds += l_ans
        preds.append(s_ans)
        questions += [question] * len(l_ans)
        questions.append(question)
        ann_texts += [ann_long_text] * len(l_ans)
        ann_texts.append(ann_short_text)
        ans_texts += ans_long_text
        ans_texts.append(ans_short_text)

        cnt += 1
        if cnt >= n_samples:
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
print(train_ann.head(10))




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
InvalidParameterError                     Traceback (most recent call last)
/tmp/ipykernel_11/3732923056.py in <cell line: 0>()
     33             ans_long_text,
     34             ans_short_text,
---> 35         ) = predict(json_data, annotated=True)
     36 
     37         ids += [f"{json_data['example_id']}_long"] * len(l_ans)

/tmp/ipykernel_11/1062082063.py in predict(json_data, annotated)
     47 
     48     tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words=stop_words)
---> 49     tfidf.fit([json_data["document_text"]])
     50     q_tfidf = tfidf.transform([question]).todense()
     51 

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

InvalidParameterError: The 'stop_words' parameter of TfidfVectorizer must be a str among {'english'}, an instance of 'list' or None. Got frozenset({'becoming', 'among', 'some', 'themselves', 'these', 'under', 'at', 'anywhere', 'which', 'often', 'cannot', 'front', 'thru', 'moreover', 'with', 'nowhere', 'him', 'though', 'first', 'mine', 'side', 'has', 'once', 'us', 'will', 'done', 'them', 'again', 'are', 'elsewhere', 'upon', 'fifty', 'hence', 'she', 'latterly', 'towards', 're', 'system', 'describe', 'another', 'other', 'into', 'off', 'full', 'been', 'from', 'would', 'less', 'across', 'thick', 'two', 'than', 'well', 'many', 'therefore', 'every', 'around', 'show', 'toward', 'sometime', 'third', 'whenever', 'somehow', 'afterwards', 'seems', 'too', 'what', 'becomes', 'more', 'wherein', 'twelve', 'get', 'against', 'interest', 'con', 'after', 'very', 'go', 'hasnt', 'anyhow', 'latter', 'serious', 'was', 'six', 'you', 'few', 'rather', 'beforehand', 'as', 'my', 'besides', 'meanwhile', 'anyone', 'empty', 'all', 'somewhere', 'ever', 'hereupon', 'couldnt', 'name', 'because', 'eg', 'is', 'over', 'everything', 'our', 'nine', 'am', 'thus', 'put', 'fill', 'whither', 'to', 'eleven', 'yours', 'otherwise', 'further', 'without', 'whereupon', 'beyond', 'hundred', 'during', 'also', 'ourselves', 'part', 'hereby', 'anything', 'between', 'whoever', 'however', 'thereupon', 'etc', 'whatever', 'perhaps', 'found', 'almost', 'least', 'about', 'take', 'hers', 'only', 'were', 'an', 'four', 'ten', 'sincere', 'cant', 'thereby', 'become', 'inc', 'both', 'now', 'together', 'then', 'none', 'bottom', 'sixty', 'mill', 'neither', 'that', 'of', 'up', 'whereby', 'whether', 'i', 'within', 'everywhere', 'either', 'co', 'due', 'hereafter', 'the', 'seeming', 'please', 'being', 'while', 'if', 'or', 'himself', 'cry', 'fire', 'here', 'forty', 'down', 'any', 'give', 'but', 'itself', 'same', 'his', 'much', 'out', 'seemed', 'enough', 'next', 'do', 'before', 'amongst', 'whose', 'became', 'although', 'de', 'each', 'thence', 'not', 'sometimes', 'see', 'in', 'might', 'by', 'alone', 'for', 'yourself', 'last', 'when', 'he', 'except', 'there', 'one', 'already', 'should', 'top', 'and', 'whereafter', 'formerly', 'mostly', 'no', 'a', 'amount', 'former', 'wherever', 'most', 'her', 'may', 'such', 'so', 'this', 'onto', 'everyone', 'ie', 'eight', 'bill', 'namely', 'detail', 'someone', 'back', 'find', 'below', 'several', 'never', 'could', 'whereas', 'since', 'thin', 'through', 'can', 'noone', 'something', 'it', 'above', 'seem', 'your', 'along', 'therein', 'even', 'they', 'nor', 'fifteen', 'book', 'nobody', 'beside', 'three', 'un', 'we', 'anyway', 'on', 'behind', 'had', 'throughout', 'its', 'be', 'where', 'myself', 'herself', 'those', 'amoungst', 'twenty', 'per', 'always', 'keep', 'who', 'how', 'herein', 'ltd', 'their', 'others', 'via', 'whole', 'yet', 'thereafter', 'still', 'nevertheless', 'whence', 'nothing', 'own', 'move', 'else', 'me', 'must', 'until', 'made', 'have', 'ours', 'whom', 'call', 'why', 'five', 'yourselves', 'indeed'}) instead.

## === cell 4
f1 = f1_score(
    train_ann["CorrectString"].values,
    train_ann["PredictionString"].values,
    average="micro",
)
print(f"F1-score: {f1:.4f}")




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2303828748.py in <cell line: 0>()
      1 # Compute micro‑F1 on the small training subset
      2 f1 = f1_score(
----> 3     train_ann["CorrectString"].values,
      4     train_ann["PredictionString"].values,
      5     average="micro",

NameError: name 'train_ann' is not defined

## === cell 5
ids = []
preds = []
questions = []
ans_texts = []

test_path = "/kaggle/input/tensorflow2-question-answering/simplified-nq-test.jsonl"
with open(test_path, "r") as json_file:
    for line in tqdm(json_file):
        json_data = json.loads(line)

        l_ans, s_ans, question, _, _, ans_long_text, ans_short_text = predict(json_data)

        ids += [f"{json_data['example_id']}_long"] * len(l_ans)
        ids.append(f"{json_data['example_id']}_short")
        preds += l_ans
        preds.append(s_ans)
        questions += [question] * len(l_ans)
        questions.append(question)
        ans_texts += ans_long_text
        ans_texts.append(ans_short_text)

subm = pd.DataFrame(
    {
        "example_id": ids,
        "question": questions,
        "PredictionString": preds,
        "PredictionText": ans_texts,
    }
)
subm.to_csv("test_data.csv", index=False)

submission = (
    subm[["example_id", "PredictionString"]]
    .groupby("example_id")
    .agg(lambda x: " ".join(x) if len(x) > 1 else x.iloc[0])
    .reset_index()
)
submission.to_csv("submission.csv", index=False)
print(submission.head(10))




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
InvalidParameterError                     Traceback (most recent call last)
/tmp/ipykernel_11/2686815868.py in <cell line: 0>()
     10         json_data = json.loads(line)
     11 
---> 12         l_ans, s_ans, question, _, _, ans_long_text, ans_short_text = predict(json_data)
     13 
     14         ids += [f"{json_data['example_id']}_long"] * len(l_ans)

/tmp/ipykernel_11/1062082063.py in predict(json_data, annotated)
     47 
     48     tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words=stop_words)
---> 49     tfidf.fit([json_data["document_text"]])
     50     q_tfidf = tfidf.transform([question]).todense()
     51 

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

InvalidParameterError: The 'stop_words' parameter of TfidfVectorizer must be a str among {'english'}, an instance of 'list' or None. Got frozenset({'becoming', 'among', 'some', 'themselves', 'these', 'under', 'at', 'anywhere', 'which', 'often', 'cannot', 'front', 'thru', 'moreover', 'with', 'nowhere', 'him', 'though', 'first', 'mine', 'side', 'has', 'once', 'us', 'will', 'done', 'them', 'again', 'are', 'elsewhere', 'upon', 'fifty', 'hence', 'she', 'latterly', 'towards', 're', 'system', 'describe', 'another', 'other', 'into', 'off', 'full', 'been', 'from', 'would', 'less', 'across', 'thick', 'two', 'than', 'well', 'many', 'therefore', 'every', 'around', 'show', 'toward', 'sometime', 'third', 'whenever', 'somehow', 'afterwards', 'seems', 'too', 'what', 'becomes', 'more', 'wherein', 'twelve', 'get', 'against', 'interest', 'con', 'after', 'very', 'go', 'hasnt', 'anyhow', 'latter', 'serious', 'was', 'six', 'you', 'few', 'rather', 'beforehand', 'as', 'my', 'besides', 'meanwhile', 'anyone', 'empty', 'all', 'somewhere', 'ever', 'hereupon', 'couldnt', 'name', 'because', 'eg', 'is', 'over', 'everything', 'our', 'nine', 'am', 'thus', 'put', 'fill', 'whither', 'to', 'eleven', 'yours', 'otherwise', 'further', 'without', 'whereupon', 'beyond', 'hundred', 'during', 'also', 'ourselves', 'part', 'hereby', 'anything', 'between', 'whoever', 'however', 'thereupon', 'etc', 'whatever', 'perhaps', 'found', 'almost', 'least', 'about', 'take', 'hers', 'only', 'were', 'an', 'four', 'ten', 'sincere', 'cant', 'thereby', 'become', 'inc', 'both', 'now', 'together', 'then', 'none', 'bottom', 'sixty', 'mill', 'neither', 'that', 'of', 'up', 'whereby', 'whether', 'i', 'within', 'everywhere', 'either', 'co', 'due', 'hereafter', 'the', 'seeming', 'please', 'being', 'while', 'if', 'or', 'himself', 'cry', 'fire', 'here', 'forty', 'down', 'any', 'give', 'but', 'itself', 'same', 'his', 'much', 'out', 'seemed', 'enough', 'next', 'do', 'before', 'amongst', 'whose', 'became', 'although', 'de', 'each', 'thence', 'not', 'sometimes', 'see', 'in', 'might', 'by', 'alone', 'for', 'yourself', 'last', 'when', 'he', 'except', 'there', 'one', 'already', 'should', 'top', 'and', 'whereafter', 'formerly', 'mostly', 'no', 'a', 'amount', 'former', 'wherever', 'most', 'her', 'may', 'such', 'so', 'this', 'onto', 'everyone', 'ie', 'eight', 'bill', 'namely', 'detail', 'someone', 'back', 'find', 'below', 'several', 'never', 'could', 'whereas', 'since', 'thin', 'through', 'can', 'noone', 'something', 'it', 'above', 'seem', 'your', 'along', 'therein', 'even', 'they', 'nor', 'fifteen', 'book', 'nobody', 'beside', 'three', 'un', 'we', 'anyway', 'on', 'behind', 'had', 'throughout', 'its', 'be', 'where', 'myself', 'herself', 'those', 'amoungst', 'twenty', 'per', 'always', 'keep', 'who', 'how', 'herein', 'ltd', 'their', 'others', 'via', 'whole', 'yet', 'thereafter', 'still', 'nevertheless', 'whence', 'nothing', 'own', 'move', 'else', 'me', 'must', 'until', 'made', 'have', 'ours', 'whom', 'call', 'why', 'five', 'yourselves', 'indeed'}) instead.

## === cell 6
print(submission.head())

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1922065815.py in <cell line: 0>()
      1 # Quick sanity check: display first few rows of the final submission
----> 2 print(submission.head())

NameError: name 'submission' is not defined
