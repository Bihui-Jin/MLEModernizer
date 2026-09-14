# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.8

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import json
import numpy as np
import pandas as pd
import re

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm_notebook as tqdm

try:
    from Levenshtein import ratio as levenshtein_distance
except ModuleNotFoundError:
    from difflib import SequenceMatcher

    def levenshtein_distance(a, b):
        return SequenceMatcher(None, a, b).ratio()


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text

from scipy import spatial


## === cell 1
n_answers = 5


## === cell 2
html_tags = ['', '', '', '', '', '', '', '', '', '', '', \
             '', '', '', '', '', '', '']
r_buf = ['is', 'are', 'do', 'does', 'did', 'was', 'were', 'will', 'can', 'the', 'a', 'of', 'in', 'and', 'on', \
         'what', 'where', 'when', 'which'] + html_tags

def clean(x):
    x = x.lower()
    for r in r_buf:
        x = x.replace(r, '')
    x = re.sub(' +', ' ', x)
    return x

bin_question_tokens = ['is', 'are', 'do', 'does', 'did', 'was', 'were', 'will', 'can']
stop_words = text.ENGLISH_STOP_WORDS.union(["book"])

def predict(json_data, annotated=False):
    candidates = json_data['long_answer_candidates']
    candidates = [c for c in candidates if c['top_level'] == True]
    doc_tokenized = json_data['document_text'].split(' ')
    question = json_data['question_text']
    question_s = question.split(' ') 
    if annotated:
        ann = json_data['annotations'][0]

    tfidf = TfidfVectorizer(ngram_range=(1,1), stop_words=stop_words)
    tfidf.fit([json_data['document_text']])
    q_tfidf = tfidf.transform([question]).todense()

    distances = []
    scores = []
    i_ann = -1
    for i, c in enumerate(candidates):
        s, e = c['start_token'], c['end_token']
        t = ' '.join(doc_tokenized[s:e])
        distances.append(levenshtein_distance(clean(question), clean(t)))
        
        t_tfidf = tfidf.transform([t]).todense()
        score = 1 - spatial.distance.cosine(q_tfidf, t_tfidf)
        
        

        scores.append(score)

    ans = (np.array(candidates)[np.argsort(scores)])[-n_answers:].tolist()
    
    if np.max(scores) < 0.2:
        ans_long = ['-1:-1']
        ans = [{'start_token': 0, 'end_token': 0}]
    else:
        ans_long = [str(a['start_token']) + ':' + str(a['end_token']) for a in ans]
    if question_s[0] in bin_question_tokens:
        ans_short = 'YES'
    else:
        ans_short = ''
        
    if annotated:
        ann_long_text = ' '.join(doc_tokenized[ann['long_answer']['start_token']:ann['long_answer']['end_token']])
        if ann['yes_no_answer'] == 'NONE':
            if len(json_data['annotations'][0]['short_answers']) > 0:
                ann_short_text = ' '.join(doc_tokenized[ann['short_answers'][0]['start_token']:ann['short_answers'][0]['end_token']])
            else:
                ann_short_text = ''
        else:
            ann_short_text = ann['yes_no_answer']
    else:
        ann_long_text = ''
        ann_short_text = ''
        
    ans_long_text = [' '.join(doc_tokenized[a['start_token']:a['end_token']]) for a in ans]
    if len(ans_short) > 0 or ans_short == 'YES':
        ans_short_text = ans_short
    else:
        ans_short_text = '' # Fix when short answers will work
                    
    return ans_long, ans_short, question, ann_long_text, ann_short_text, ans_long_text, ans_short_text


## === cell 3
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
stop_words = list(text.ENGLISH_STOP_WORDS.union(["book"]))


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
    i_ann = -1
    for i, c in enumerate(candidates):
        s, e = c["start_token"], c["end_token"]
        t = " ".join(doc_tokenized[s:e])
        distances.append(levenshtein_distance(clean(question), clean(t)))

        t_tfidf = tfidf.transform([t]).todense()
        score = 1 - spatial.distance.cosine(q_tfidf, t_tfidf)

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
    if len(ans_short) > 0 or ans_short == "YES":
        ans_short_text = ans_short
    else:
        ans_short_text = ""  # Fix when short answers will work

    return (
        ans_long,
        ans_short,
        question,
        ann_long_text,
        ann_short_text,
        ans_long_text,
        ans_short_text,
    )


## === cell 4
if (
    "train_ann" in globals()
    and isinstance(train_ann, pd.DataFrame)
    and {"CorrectString", "PredictionString"}.issubset(train_ann.columns)
):
    f1 = f1_score(
        train_ann["CorrectString"].values,
        train_ann["PredictionString"].values,
        average="micro",
    )
    print(f"F1-score: {f1:.4f}")
else:
    print("Skipping F1-score: `train_ann` not defined (or missing required columns).")


## === cell 5
%%time
ids = []
anns = []
preds = []

questions = []
ann_texts = []
ans_texts = []

with open('/kaggle/input/tensorflow2-question-answering/simplified-nq-test.jsonl', 'r') as json_file:
    cnt = 0
    for line in tqdm(json_file):
        json_data = json.loads(line)
        
        l_ans, s_ans, question, ann_long_text, ann_short_text, ans_long_text, ans_short_text = predict(json_data)

        ids += [str(json_data['example_id']) + '_long']*len(l_ans)
        ids.append(str(json_data['example_id']) + '_short')
        preds += l_ans
        preds.append(s_ans)
        questions += [question]*len(l_ans)
        questions.append(question)
        ans_texts += ans_long_text
        ans_texts.append(ans_short_text)
         
        
subm = pd.DataFrame()
subm['example_id'] = ids
subm['question'] = questions
subm['PredictionString'] = preds
subm['PredictionText'] = ans_texts
subm.to_csv('test_data.csv', index=False)

g = subm[['example_id', 'PredictionString']].groupby('example_id').agg(lambda x: '[' + ', '.join(x) + ']' if len(x) > 1 else x).reset_index()
g.to_csv('submission.csv', index=False)

subm.head(10)


## --- ERROR in cell 5, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m<timed exec>[0m in [0;36m<module>[0;34m[0m

[0;32m/tmp/ipykernel_11/3642033665.py[0m in [0;36mpredict[0;34m(json_data, annotated)[0m
[1;32m     59[0m [0;34m[0m[0m
[1;32m     60[0m         [0mt_tfidf[0m [0;34m=[0m [0mtfidf[0m[0;34m.[0m[0mtransform[0m[0;34m([0m[0;34m[[0m[0mt[0m[0;34m][0m[0;34m)[0m[0;34m.[0m[0mtodense[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 61[0;31m         [0mscore[0m [0;34m=[0m [0;36m1[0m [0;34m-[0m [0mspatial[0m[0;34m.[0m[0mdistance[0m[0;34m.[0m[0mcosine[0m[0;34m([0m[0mq_tfidf[0m[0;34m,[0m [0mt_tfidf[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     62[0m [0;34m[0m[0m
[1;32m     63[0m         [0mscores[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0mscore[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/scipy/spatial/distance.py[0m in [0;36mcosine[0;34m(u, v, w)[0m
[1;32m    736[0m     [0;31m# cosine distance is also referred to as 'uncentered correlation',[0m[0;34m[0m[0;34m[0m[0m
[1;32m    737[0m     [0;31m#   or 'reflective correlation'[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 738[0;31m     [0;32mreturn[0m [0mcorrelation[0m[0;34m([0m[0mu[0m[0;34m,[0m [0mv[0m[0;34m,[0m [0mw[0m[0;34m=[0m[0mw[0m[0;34m,[0m [0mcentered[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    739[0m [0;34m[0m[0m
[1;32m    740[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/scipy/spatial/distance.py[0m in [0;36mcorrelation[0;34m(u, v, w, centered)[0m
[1;32m    656[0m     [0;36m0.5[0m[0;34m[0m[0;34m[0m[0m
[1;32m    657[0m     """
[0;32m--> 658[0;31m     [0mu[0m [0;34m=[0m [0m_validate_vector[0m[0;34m([0m[0mu[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    659[0m     [0mv[0m [0;34m=[0m [0m_validate_vector[0m[0;34m([0m[0mv[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    660[0m     [0;32mif[0m [0mnp[0m[0;34m.[0m[0miscomplexobj[0m[0;34m([0m[0mu[0m[0;34m)[0m [0;32mor[0m [0mnp[0m[0;34m.[0m[0miscomplexobj[0m[0;34m([0m[0mv[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/scipy/spatial/distance.py[0m in [0;36m_validate_vector[0;34m(u, dtype)[0m
[1;32m    300[0m     [0;32mif[0m [0mu[0m[0;34m.[0m[0mndim[0m [0;34m==[0m [0;36m1[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    301[0m         [0;32mreturn[0m [0mu[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 302[0;31m     [0;32mraise[0m [0mValueError[0m[0;34m([0m[0;34m"Input vector should be 1-D."[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    303[0m [0;34m[0m[0m
[1;32m    304[0m [0;34m[0m[0m

[0;31mValueError[0m: Input vector should be 1-D.

## === cell 6
g.head()
