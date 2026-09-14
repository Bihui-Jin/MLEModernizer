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

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm_notebook as tqdm

try:
    from Levenshtein import ratio as levenshtein_distance  # type: ignore
except ModuleNotFoundError:
    from difflib import SequenceMatcher

    def levenshtein_distance(a, b):
        return SequenceMatcher(None, a, b).ratio()


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text

from scipy import spatial


## === cell 1
r_buf = ['is', 'are', 'do', 'does', 'did', 'was', 'were', 'will', 'can', 'the', 'a', 'of', 'in', 'and', 'on', \
         'what', 'where', 'when', 'which']
def clean(x):
    x = x.lower()
    for r in r_buf:
        x = x.replace(r, '')
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
        distances.append(1 - levenshtein_distance(clean(question), clean(t)))
        
        t_tfidf = tfidf.transform([t]).todense()
        score = 1 - spatial.distance.cosine(q_tfidf, t_tfidf)
        
        

        scores.append(score)

    ans = candidates[np.argmax(scores)]
    ans_long = str(ans['start_token']) + ':' + str(ans['end_token'])
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
        
    ans_long_text = ' '.join(doc_tokenized[ans['start_token']:ans['end_token']])
    if len(ans_short) > 0 or ans_short == 'YES':
        ans_short_text = ans_short
    else:
        ans_short_text = '' # Fix when short answers will work
            
    return ans_long, ans_short, question, ann_long_text, ann_short_text, ans_long_text, ans_short_text


## === cell 2
%%time
ids = []
anns = []
preds = []

questions = []
ann_texts = []
ans_texts = []

n_samples = 500

with open('/kaggle/input/tensorflow2-question-answering/simplified-nq-train.jsonl', 'r') as json_file:
    cnt = 0
    for line in tqdm(json_file):
        json_data = json.loads(line)
        
        ids.append(str(json_data['example_id']) + '_long')
        ids.append(str(json_data['example_id']) + '_short')
        
        l_ans = str(json_data['annotations'][0]['long_answer']['start_token']) + ':' + \
            str(json_data['annotations'][0]['long_answer']['end_token'])
        if json_data['annotations'][0]['yes_no_answer'] == 'NONE':
            if len(json_data['annotations'][0]['short_answers']) > 0:
                s_ans = str(json_data['annotations'][0]['short_answers'][0]['start_token']) + ':' + \
                    str(json_data['annotations'][0]['short_answers'][0]['end_token'])
            else:
                s_ans = ''
        else:
            s_ans = json_data['annotations'][0]['yes_no_answer']
            
        anns.append(l_ans)
        anns.append(s_ans)
        
        l_ans, s_ans, question, ann_long_text, ann_short_text, ans_long_text, ans_short_text = predict(json_data, annotated=True)
        
        preds.append(l_ans)
        preds.append(s_ans)
        questions.append(question)
        questions.append(question)
        ann_texts.append(ann_long_text)
        ann_texts.append(ann_short_text)
        ans_texts.append(ans_long_text)
        ans_texts.append(ans_short_text)
        
        cnt += 1
        if cnt >= n_samples:
            break
        
train_ann = pd.DataFrame()
train_ann['example_id'] = ids
train_ann['question'] = questions
train_ann['CorrectString'] = anns
train_ann['CorrectText'] = ann_texts
if len(preds) > 0:
    train_ann['PredictionString'] = preds
    train_ann['PredictionText'] = ans_texts
    
train_ann.to_csv('train_data.csv', index=False)
train_ann.head(10)


## --- ERROR in cell 2, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mInvalidParameterError[0m                     Traceback (most recent call last)
[0;32m<timed exec>[0m in [0;36m<module>[0;34m[0m

[0;32m/tmp/ipykernel_11/2171180564.py[0m in [0;36mpredict[0;34m(json_data, annotated)[0m
[1;32m     22[0m     [0;31m# TFIDF for the document[0m[0;34m[0m[0;34m[0m[0m
[1;32m     23[0m     [0mtfidf[0m [0;34m=[0m [0mTfidfVectorizer[0m[0;34m([0m[0mngram_range[0m[0;34m=[0m[0;34m([0m[0;36m1[0m[0;34m,[0m[0;36m1[0m[0;34m)[0m[0;34m,[0m [0mstop_words[0m[0;34m=[0m[0mstop_words[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 24[0;31m     [0mtfidf[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0;34m[[0m[0mjson_data[0m[0;34m[[0m[0;34m'document_text'[0m[0;34m][0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     25[0m     [0mq_tfidf[0m [0;34m=[0m [0mtfidf[0m[0;34m.[0m[0mtransform[0m[0;34m([0m[0;34m[[0m[0mquestion[0m[0;34m][0m[0;34m)[0m[0;34m.[0m[0mtodense[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     26[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/feature_extraction/text.py[0m in [0;36mfit[0;34m(self, raw_documents, y)[0m
[1;32m   2092[0m             [0mFitted[0m [0mvectorizer[0m[0;34m.[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2093[0m         """
[0;32m-> 2094[0;31m         [0mself[0m[0;34m.[0m[0m_validate_params[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   2095[0m         [0mself[0m[0;34m.[0m[0m_check_params[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2096[0m         [0mself[0m[0;34m.[0m[0m_warn_for_unused_params[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/base.py[0m in [0;36m_validate_params[0;34m(self)[0m
[1;32m    598[0m         [0maccepted[0m [0mconstraints[0m[0;34m.[0m[0;34m[0m[0;34m[0m[0m
[1;32m    599[0m         """
[0;32m--> 600[0;31m         validate_parameter_constraints(
[0m[1;32m    601[0m             [0mself[0m[0;34m.[0m[0m_parameter_constraints[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    602[0m             [0mself[0m[0;34m.[0m[0mget_params[0m[0;34m([0m[0mdeep[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/_param_validation.py[0m in [0;36mvalidate_parameter_constraints[0;34m(parameter_constraints, params, caller_name)[0m
[1;32m     95[0m                 )
[1;32m     96[0m [0;34m[0m[0m
[0;32m---> 97[0;31m             raise InvalidParameterError(
[0m[1;32m     98[0m                 [0;34mf"The {param_name!r} parameter of {caller_name} must be"[0m[0;34m[0m[0;34m[0m[0m
[1;32m     99[0m                 [0;34mf" {constraints_str}. Got {param_val!r} instead."[0m[0;34m[0m[0;34m[0m[0m

[0;31mInvalidParameterError[0m: The 'stop_words' parameter of TfidfVectorizer must be a str among {'english'}, an instance of 'list' or None. Got frozenset({'his', 'whenever', 'indeed', 'con', 'another', 'mostly', 'their', 'bill', 'in', 'nowhere', 'above', 'thus', 'whom', 'find', 'seeming', 'though', 'beforehand', 'see', 'without', 'therein', 'never', 'he', 'yet', 'system', 'against', 'alone', 'been', 'at', 'fifty', 're', 'anywhere', 'it', 'whereafter', 'because', 'get', 'were', 'out', 'an', 'why', 'hundred', 'would', 'whatever', 'sincere', 'sometime', 'onto', 'once', 'around', 'also', 'each', 'therefore', 'none', 'was', 'rather', 'first', 'moreover', 'please', 'we', 'down', 'whereby', 'any', 'afterwards', 'ltd', 'fifteen', 'the', 'now', 'twenty', 'such', 'least', 'enough', 'together', 'side', 'some', 'bottom', 'eg', 'amongst', 'everything', 'amount', 'all', 'thru', 'is', 'anyway', 'seemed', 'per', 'call', 'full', 'own', 'wherein', 'seems', 'several', 'namely', 'elsewhere', 'whoever', 'both', 'if', 'someone', 'five', 'from', 'last', 'into', 'your', 'cant', 'part', 'ours', 'eleven', 'ie', 'besides', 'meanwhile', 'etc', 'again', 'up', 'yours', 'co', 'fire', 'already', 'else', 'a', 'amoungst', 'yourself', 'well', 'nobody', 'hence', 'upon', 'not', 'off', 'eight', 'with', 'always', 'so', 'these', 'somewhere', 'many', 'could', 'before', 'whither', 'mine', 'front', 'since', 'where', 'everyone', 'more', 'through', 'after', 'yourselves', 'thereafter', 'be', 'most', 'put', 'seem', 'describe', 'as', 'even', 'but', 'whereas', 'themselves', 'our', 'made', 'same', 'i', 'serious', 'may', 'within', 'thick', 'towards', 'behind', 'me', 'when', 'nevertheless', 'are', 'which', 'either', 'or', 'cry', 'ever', 'hereafter', 'beyond', 'two', 'only', 'almost', 'here', 'can', 'much', 'thereupon', 'had', 'done', 'itself', 'below', 'than', 'this', 'everywhere', 'how', 'few', 'four', 'fill', 'has', 'whole', 'de', 'they', 'of', 'move', 'do', 'take', 'book', 'throughout', 'becomes', 'other', 'those', 'sixty', 'hasnt', 'hereupon', 'former', 'whereupon', 'hereby', 'latter', 'anyhow', 'due', 'along', 'found', 'too', 'am', 'thence', 'un', 'no', 'anyone', 'become', 'what', 'him', 'next', 'however', 'being', 'its', 'anything', 'toward', 'name', 'over', 'thin', 'via', 'you', 'wherever', 'beside', 'nor', 'between', 'ten', 'must', 'top', 'across', 'she', 'should', 'one', 'myself', 'third', 'still', 'during', 'my', 'himself', 'will', 'thereby', 'less', 'then', 'back', 'herein', 'cannot', 'often', 'became', 'keep', 'six', 'further', 'detail', 'under', 'latterly', 'formerly', 'whether', 'about', 'neither', 'herself', 'until', 'becoming', 'while', 'us', 'by', 'otherwise', 'have', 'show', 'noone', 'who', 'to', 'her', 'interest', 'hers', 'empty', 'nine', 'go', 'that', 'ourselves', 'somehow', 'three', 'something', 'for', 'couldnt', 'sometimes', 'and', 'although', 'whose', 'mill', 'might', 'give', 'others', 'inc', 'except', 'among', 'whence', 'forty', 'twelve', 'them', 'perhaps', 'very', 'nothing', 'on', 'every', 'there'}) instead.

## === cell 3
f1 = f1_score(train_ann['CorrectString'].values, train_ann['PredictionString'].values, average='micro')
print(f'F1-score: {f1:.4f}')
