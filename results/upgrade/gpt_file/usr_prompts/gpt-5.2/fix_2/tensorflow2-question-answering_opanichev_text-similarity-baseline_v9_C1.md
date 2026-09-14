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

0.00805

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
from difflib import SequenceMatcher

from sklearn.metrics import f1_score
from tqdm import tqdm


def levenshtein_distance(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from scipy import spatial

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))



## === cell 1
n_answers = 5



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
    x = x.lower()
    for r in r_buf:
        x = x.replace(r, "")
    x = re.sub(" +", " ", x)
    return x


bin_question_tokens = ["is", "are", "do", "does", "did", "was", "were", "will", "can"]
stop_words = text.ENGLISH_STOP_WORDS.union(["book"])


def predict(json_data, annotated=False):
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
    for i, c in enumerate(candidates):
        s, e = c["start_token"], c["end_token"]
        t = " ".join(doc_tokenized[s:e])
        distances.append(levenshtein_distance(clean(question), clean(t)))

        t_tfidf = tfidf.transform([t]).todense()
        score = 1 - spatial.distance.cosine(q_tfidf, t_tfidf)
        if np.isnan(score):
            score = 0.0
        scores.append(score)

    if len(candidates) == 0:
        ans = [{"start_token": 0, "end_token": 0}]
        scores = [0.0]
    else:
        ans = (np.array(candidates)[np.argsort(scores)])[-n_answers:].tolist()

    if np.max(scores) < 0.2:
        ans_long = ["-1:-1"]
        ans = [{"start_token": 0, "end_token": 0}]
    else:
        ans_long = [str(a["start_token"]) + ":" + str(a["end_token"]) for a in ans]

    if len(question_s) > 0 and question_s[0].lower() in bin_question_tokens:
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
    if len(ans_short) > 0 or ans_short == "YES":
        ans_short_text = ans_short
    else:
        ans_short_text = ""

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

n_samples = 200  # keep small for runtime; only sanity check, not used for submission

train_path = "/kaggle/input/tensorflow2-question-answering/simplified-nq-train.jsonl"
if not os.path.exists(train_path):
    train_path = "/kaggle/input/simplified-nq-train.jsonl"

with open(train_path, "r") as json_file:
    cnt = 0
    for line in tqdm(json_file, total=n_samples):
        json_data = json.loads(line)

        l_ann = (
            str(json_data["annotations"][0]["long_answer"]["start_token"])
            + ":"
            + str(json_data["annotations"][0]["long_answer"]["end_token"])
        )
        if json_data["annotations"][0]["yes_no_answer"] == "NONE":
            if len(json_data["annotations"][0]["short_answers"]) > 0:
                s_ann = (
                    str(json_data["annotations"][0]["short_answers"][0]["start_token"])
                    + ":"
                    + str(json_data["annotations"][0]["short_answers"][0]["end_token"])
                )
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

        ids += [str(json_data["example_id"]) + "_long"] * len(l_ans)
        ids.append(str(json_data["example_id"]) + "_short")

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
train_ann.head(10)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
InvalidParameterError                     Traceback (most recent call last)
/tmp/ipykernel_11/3345337550.py in <cell line: 0>()
     44             ans_long_text,
     45             ans_short_text,
---> 46         ) = predict(json_data, annotated=True)
     47 
     48         ids += [str(json_data["example_id"]) + "_long"] * len(l_ans)

/tmp/ipykernel_11/3886351459.py in predict(json_data, annotated)
     45 
     46     tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words=stop_words)
---> 47     tfidf.fit([json_data["document_text"]])
     48     q_tfidf = tfidf.transform([question]).todense()
     49 

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

InvalidParameterError: The 'stop_words' parameter of TfidfVectorizer must be a str among {'english'}, an instance of 'list' or None. Got frozenset({'what', 'very', 'will', 'without', 'ie', 'while', 'wherein', 'whereafter', 'via', 'two', 'with', 'side', 'alone', 'at', 'has', 'forty', 'ours', 'eleven', 'therefore', 'how', 'thereafter', 'i', 'whereas', 'thence', 'call', 'yet', 'because', 'a', 'still', 'beforehand', 'third', 'yourselves', 'whither', 'above', 'noone', 'any', 'such', 'etc', 'amongst', 'give', 'never', 'itself', 'eg', 'six', 'elsewhere', 'mine', 'however', 'three', 'whenever', 'often', 'then', 'that', 'system', 'among', 'along', 'others', 'last', 'to', 'afterwards', 'moreover', 'fire', 'him', 'an', 'being', 'thick', 'book', 'sometimes', 'otherwise', 'therein', 'whole', 'perhaps', 'towards', 'against', 'anywhere', 'bill', 'former', 'who', 'herein', 'co', 'we', 'wherever', 'also', 'were', 'those', 'but', 'empty', 'hundred', 'is', 'my', 'meanwhile', 'something', 'fifteen', 'ltd', 'becomes', 'have', 'full', 'may', 'each', 'only', 'became', 'least', 'so', 'whereupon', 'thereby', 'you', 'most', 'further', 'her', 'us', 'thereupon', 'them', 'where', 'upon', 'around', 'un', 'himself', 'thus', 'sometime', 'he', 'else', 'nor', 'many', 'latterly', 'made', 'some', 'becoming', 'hereupon', 'describe', 're', 'indeed', 'whence', 'see', 'off', 'yourself', 'formerly', 'was', 'de', 'every', 'both', 'for', 'seems', 'no', 'behind', 'ourselves', 'might', 'after', 'and', 'been', 'amount', 'or', 'five', 'although', 'herself', 'seemed', 'his', 'myself', 'before', 'anyhow', 'beyond', 'since', 'ten', 'serious', 'on', 'throughout', 'which', 'anything', 'between', 'cannot', 'would', 'across', 'seem', 'couldnt', 'in', 'inc', 'whereby', 'anyway', 'the', 'latter', 'nobody', 'someone', 'top', 'namely', 'another', 'whoever', 'me', 'cant', 'why', 'cry', 'move', 'together', 'same', 'it', 'ever', 'nothing', 'amoungst', 'almost', 'well', 'fifty', 'name', 'put', 'detail', 'even', 'down', 'much', 'none', 'here', 'keep', 'by', 'whose', 'thru', 'become', 'be', 'one', 'mostly', 'seeming', 'our', 'they', 'several', 'first', 'part', 'until', 'hasnt', 'over', 'always', 'your', 'beside', 'hereafter', 'nowhere', 'mill', 'hence', 'am', 'under', 'get', 'due', 'eight', 'everyone', 'yours', 'sixty', 'twelve', 'hers', 'could', 'up', 'own', 'whatever', 'too', 'rather', 'below', 'should', 'per', 'anyone', 'themselves', 'con', 'everything', 'about', 'all', 'done', 'whom', 'neither', 'except', 'these', 'of', 'other', 'its', 'though', 'now', 'four', 'as', 'this', 'next', 'nine', 'take', 'nevertheless', 'back', 'she', 'fill', 'enough', 'had', 'somewhere', 'please', 'already', 'hereby', 'onto', 'out', 'can', 'from', 'not', 'somehow', 'there', 'interest', 'do', 'sincere', 'whether', 'within', 'find', 'few', 'again', 'their', 'bottom', 'found', 'everywhere', 'are', 'once', 'go', 'must', 'into', 'twenty', 'either', 'through', 'front', 'show', 'toward', 'besides', 'less', 'more', 'if', 'than', 'when', 'during', 'thin'}) instead.

## === cell 4
try:
    f1 = f1_score(
        train_ann["CorrectString"].values,
        train_ann["PredictionString"].values,
        average="micro",
    )
    print(f"F1-score (string micro, debug only): {f1:.4f}")
except Exception as e:
    print("Could not compute debug F1:", repr(e))



## === cell 5
test_path = "/kaggle/input/tensorflow2-question-answering/simplified-nq-test.jsonl"
if not os.path.exists(test_path):
    test_path = "/kaggle/input/simplified-nq-test.jsonl"

ids = []
preds = []
questions = []
ans_texts = []

with open(test_path, "r") as json_file:
    for line in tqdm(json_file):
        json_data = json.loads(line)

        l_ans, s_ans, question, _, _, ans_long_text, ans_short_text = predict(
            json_data, annotated=False
        )

        ids += [str(json_data["example_id"]) + "_long"] * len(l_ans)
        preds += l_ans
        questions += [question] * len(l_ans)
        ans_texts += ans_long_text

        ids.append(str(json_data["example_id"]) + "_short")
        preds.append(s_ans)
        questions.append(question)
        ans_texts.append(ans_short_text)

test_df = pd.DataFrame(
    {
        "example_id": ids,
        "question": questions,
        "PredictionString": preds,
        "PredictionText": ans_texts,
    }
)
test_df.to_csv("test_data.csv", index=False)

sample_path = "/kaggle/input/tensorflow2-question-answering/sample_submission.csv"
if not os.path.exists(sample_path):
    sample_path = "/kaggle/input/sample_submission.csv"
sample_sub = pd.read_csv(sample_path)

pred_map = test_df.groupby("example_id")["PredictionString"].last()

submission = sample_sub.copy()
submission["PredictionString"] = submission["example_id"].map(pred_map)

submission["PredictionString"] = submission["PredictionString"].fillna("")

submission.to_csv("submission.csv", index=False)
print(submission.head(10))
print("Wrote submission.csv with shape:", submission.shape)

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
InvalidParameterError                     Traceback (most recent call last)
/tmp/ipykernel_11/1037461259.py in <cell line: 0>()
     13         json_data = json.loads(line)
     14 
---> 15         l_ans, s_ans, question, _, _, ans_long_text, ans_short_text = predict(
     16             json_data, annotated=False
     17         )

/tmp/ipykernel_11/3886351459.py in predict(json_data, annotated)
     45 
     46     tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words=stop_words)
---> 47     tfidf.fit([json_data["document_text"]])
     48     q_tfidf = tfidf.transform([question]).todense()
     49 

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

InvalidParameterError: The 'stop_words' parameter of TfidfVectorizer must be a str among {'english'}, an instance of 'list' or None. Got frozenset({'what', 'very', 'will', 'without', 'ie', 'while', 'wherein', 'whereafter', 'via', 'two', 'with', 'side', 'alone', 'at', 'has', 'forty', 'ours', 'eleven', 'therefore', 'how', 'thereafter', 'i', 'whereas', 'thence', 'call', 'yet', 'because', 'a', 'still', 'beforehand', 'third', 'yourselves', 'whither', 'above', 'noone', 'any', 'such', 'etc', 'amongst', 'give', 'never', 'itself', 'eg', 'six', 'elsewhere', 'mine', 'however', 'three', 'whenever', 'often', 'then', 'that', 'system', 'among', 'along', 'others', 'last', 'to', 'afterwards', 'moreover', 'fire', 'him', 'an', 'being', 'thick', 'book', 'sometimes', 'otherwise', 'therein', 'whole', 'perhaps', 'towards', 'against', 'anywhere', 'bill', 'former', 'who', 'herein', 'co', 'we', 'wherever', 'also', 'were', 'those', 'but', 'empty', 'hundred', 'is', 'my', 'meanwhile', 'something', 'fifteen', 'ltd', 'becomes', 'have', 'full', 'may', 'each', 'only', 'became', 'least', 'so', 'whereupon', 'thereby', 'you', 'most', 'further', 'her', 'us', 'thereupon', 'them', 'where', 'upon', 'around', 'un', 'himself', 'thus', 'sometime', 'he', 'else', 'nor', 'many', 'latterly', 'made', 'some', 'becoming', 'hereupon', 'describe', 're', 'indeed', 'whence', 'see', 'off', 'yourself', 'formerly', 'was', 'de', 'every', 'both', 'for', 'seems', 'no', 'behind', 'ourselves', 'might', 'after', 'and', 'been', 'amount', 'or', 'five', 'although', 'herself', 'seemed', 'his', 'myself', 'before', 'anyhow', 'beyond', 'since', 'ten', 'serious', 'on', 'throughout', 'which', 'anything', 'between', 'cannot', 'would', 'across', 'seem', 'couldnt', 'in', 'inc', 'whereby', 'anyway', 'the', 'latter', 'nobody', 'someone', 'top', 'namely', 'another', 'whoever', 'me', 'cant', 'why', 'cry', 'move', 'together', 'same', 'it', 'ever', 'nothing', 'amoungst', 'almost', 'well', 'fifty', 'name', 'put', 'detail', 'even', 'down', 'much', 'none', 'here', 'keep', 'by', 'whose', 'thru', 'become', 'be', 'one', 'mostly', 'seeming', 'our', 'they', 'several', 'first', 'part', 'until', 'hasnt', 'over', 'always', 'your', 'beside', 'hereafter', 'nowhere', 'mill', 'hence', 'am', 'under', 'get', 'due', 'eight', 'everyone', 'yours', 'sixty', 'twelve', 'hers', 'could', 'up', 'own', 'whatever', 'too', 'rather', 'below', 'should', 'per', 'anyone', 'themselves', 'con', 'everything', 'about', 'all', 'done', 'whom', 'neither', 'except', 'these', 'of', 'other', 'its', 'though', 'now', 'four', 'as', 'this', 'next', 'nine', 'take', 'nevertheless', 'back', 'she', 'fill', 'enough', 'had', 'somewhere', 'please', 'already', 'hereby', 'onto', 'out', 'can', 'from', 'not', 'somehow', 'there', 'interest', 'do', 'sincere', 'whether', 'within', 'find', 'few', 'again', 'their', 'bottom', 'found', 'everywhere', 'are', 'once', 'go', 'must', 'into', 'twenty', 'either', 'through', 'front', 'show', 'toward', 'besides', 'less', 'more', 'if', 'than', 'when', 'during', 'thin'}) instead.
