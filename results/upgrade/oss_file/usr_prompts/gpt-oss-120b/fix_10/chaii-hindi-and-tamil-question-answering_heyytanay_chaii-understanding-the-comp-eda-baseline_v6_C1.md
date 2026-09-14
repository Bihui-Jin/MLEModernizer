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
Predicting the answers to questions in Hindi and Tamil.

## Metric
Word-level Jaccard score.

A Python implementation is provided below.

```
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
```

The formula for the overall metric is:
\text{score} = \frac{1}{n} \sum_{i=1}^n \text{jaccard}(gt_i, dt_i)

where:
$n$ = number of documents

$\text{jaccard}$ = the function provided above

$gt_i$ = the ith ground truth

$dt_i$ = the ith prediction

## Submission Format
For each ID in the test set, you must predict the string that best answers the provided question based on the context. Note that the selected text needs to be quoted and complete to work correctly. Include punctuation, etc. The file should contain a header and have the following format:

```
id,PredictionString
8c8ee6504,"1"
3163c22d0,"2 string"
66aae423b,"4 word 6"
722085a7b,"1"
etc.
```

## Dataset 
**All files should be encoded as UTF-8.**

- **train.csv** - the training set, containing context, questions, and answers. Also includes the start character of the answer for disambiguation.
- **test.csv** - the test set, containing context and questions.
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `id` - a unique identifier
- `context` - the text of the Hindi/Tamil sample from which answers should be derived
- `question` - the question, in Hindi/Tamil
- `answer_text` (train only) - the answer to the question (manual annotation) (note: for test, this is what you are attempting to predict)
- `answer_start` (train only) - the starting character in `context` for the answer (determined using substring match during data preparation)
- `language` - whether the text in question is in Tamil or Hindi

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
plotly==5.24.1
plotly-express==0.4.1
rich==14.2.0
seaborn==0.12.2
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
transformers==4.53.3
wordcloud==1.9.4

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (138 lines)
            sample_submission.csv (113 lines)
            sample_submission.csv.zip (950 Bytes)
            test.csv (7173 lines)
            test.csv.zip (648.4 kB)
            train.csv (67723 lines)
            train.csv.zip (5.9 MB)
            chaii-hindi-and-tamil-question-answering/
                description.md (138 lines)
                sample_submission.csv (113 lines)
                ... and 5 other files
                chaii-hindi-and-tamil-question-answering/
        input/
            description.md (138 lines)
            sample_submission.csv (113 lines)
            sample_submission.csv.zip (950 Bytes)
            test.csv (7173 lines)
            test.csv.zip (648.4 kB)
            train.csv (67723 lines)
            train.csv.zip (5.9 MB)
            chaii-hindi-and-tamil-question-answering/
                description.md (138 lines)
                sample_submission.csv (113 lines)
                ... and 5 other files
                chaii-hindi-and-tamil-question-answering/
        working/
            chaii-hindi-and-tamil-question-answering/
                description.md (138 lines)
                sample_submission.csv (113 lines)
                ... and 5 other files
                chaii-hindi-and-tamil-question-answering/
```

-> data/chaii-hindi-and-tamil-question-answering/sample_submission.csv has 112 rows and 2 columns.
The columns are: id, PredictionString

-> data/chaii-hindi-and-tamil-question-answering/test.csv has 7172 rows and 4 columns.
The columns are: id, context, question, language

-> data/chaii-hindi-and-tamil-question-answering/train.csv has 67722 rows and 6 columns.
The columns are: id, context, question, answer_text, answer_start, language

-> data/sample_submission.csv has 112 rows and 2 columns.
The columns are: id, PredictionString

-> data/test.csv has 7172 rows and 4 columns.
The columns are: id, context, question, language

-> data/train.csv has 67722 rows and 6 columns.
The columns are: id, context, question, answer_text, answer_start, language

-> input/chaii-hindi-and-tamil-question-answering/sample_submission.csv has 112 rows and 2 columns.
The columns are: id, PredictionString

-> (stopped after 10 files for performance)

# 5. Target score

0.4375553727149963

# 6. Current score

0.07282

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'Implemented minimal fixes to unblock execution and generate a proper submission:

1. Wrapped optional visualization imports in a safe try/except block to avoid protobuf‑related import errors.  
2. Corrected the model directory path to the absolute Kaggle input location (`/kaggle/input/...`) and kept `local_files_only=True` so the model loads from the local cache without hub validation.  
3. Added a fallback to skip loading the model if the directory is missing, preventing crashes.  
4. Ensured the predictions list is defined before creating the submission file.'
- What this solution (achieved 0.05152) has done: 'I replace the failing model‑loading block with a safe rule‑based fallback: first try to reuse an exact‑match answer from the training set, and if none exists, use a short snippet of the context. This removes the protobuf error, ensures a non‑empty prediction list, and should raise the Jaccard score above 0.0 while preserving the original workflow.'
- What this solution (achieved 0.01731) has done: 'I tighten the rule‑based inference: match questions case‑insensitively and, when a question is unseen, fall back to the first full sentence (or the first 30 characters) of the context instead of an arbitrary short snippet. This small tweak adds more realistic answer text and should raise the Jaccard score toward the target while keeping the original pipeline intact.'
- What this solution (achieved 0.04293) has done: 'I enhance the rule‑based inference by adding a lightweight fuzzy‑matching fallback: for each test question we first try an exact lowercase match, then look for a training question that shares the same first 5 characters and has a high string‑similarity score (≥ 0.8). If such a close match is found we use its answer; otherwise we keep the previous sentence‑fallback. This small improvement should raise the Jaccard score toward the target while preserving the original pipeline.'
- What this solution (achieved 0.06472) has done: 'I broaden the fuzzy‑matching step so that, after attempting an exact lookup, the code compares the test question against *all* training questions (instead of only those sharing the same prefix). If the best similarity ratio reaches a lower threshold (0.6), the corresponding answer is used. This keeps the overall rule‑based workflow intact while providing many more useful matches, which should raise the Jaccard score toward the target.'
- What this solution (achieved 0.06567) has done: 'I improve the rule‑based inference by (1) lowering the fuzzy‑match threshold slightly so more useful matches are accepted, and (2) selecting a fallback sentence that shares the most words with the question instead of always taking the first sentence. These tweaks keep the original workflow intact while providing answers that better overlap the ground‑truth, which should raise the Jaccard score toward the target.'
- What this solution (achieved 0.07381) has done: 'I lower the fuzzy‑matching acceptance threshold (to capture more useful training answers) and make the sentence fallback a bit tighter by truncating overly long sentences, which should increase the overlap with the true answers and therefore raise the Jaccard score toward the target.'
- What this solution (achieved 0.07282) has done: 'I add a lightweight TF‑IDF similarity lookup to retrieve answers from the most similar training question, lower the fuzzy‑match thresholds slightly to accept more matches, and integrate this step into the prediction loop before the sentence fallback. These changes keep the original rule‑based workflow while giving the model more chances to output a relevant answer, which should raise the Jaccard score toward the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import difflib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.graph_objs as go
    import plotly.figure_factory as ff
    import plotly.express as px
    from plotly.subplots import make_subplots
    from plotly.offline import iplot
    from wordcloud import WordCloud
    from rich import print as _pprint
except Exception:

    def _pprint(*args, **kwargs):
        print(*args, **kwargs)




## === cell 1
def cprint(string):
    """
    Utility function for beautiful colored printing.
    """
    _pprint(f"[black]{string}[/black]")




## === cell 2
train_file = pd.read_csv("../input/chaii-hindi-and-tamil-question-answering/train.csv")
test_file = pd.read_csv("../input/chaii-hindi-and-tamil-question-answering/test.csv")
sample_sub = pd.read_csv(
    "../input/chaii-hindi-and-tamil-question-answering/sample_submission.csv"
)

train_questions = train_file["question"].astype(str).str.lower().tolist()
tfidf_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5))
tfidf_matrix = tfidf_vectorizer.fit_transform(train_questions)
train_answers = train_file["answer_text"].astype(str).tolist()


def get_answer_by_tfidf(query, min_score=0.15):
    """Return the answer of the most similar training question if similarity exceeds min_score."""
    query_vec = tfidf_vectorizer.transform([query.lower()])
    sims = linear_kernel(query_vec, tfidf_matrix).flatten()
    best_idx = sims.argmax()
    if sims[best_idx] >= min_score:
        return train_answers[best_idx]
    return None




## === cell 3
train_file.head()




## === cell 4
train_file.info()




## === cell 5
train_file.describe()




## === cell 6
test_file.head()




## === cell 7
test_file.info()




## === cell 8
test_file.describe()




## === cell 9
sample_sub.head()




## === cell 10
cprint("Total Training Examples: [green]{}[/green]".format(train_file.shape[0]))
cprint("Total Testing Examples: [green]{}[/green]".format(test_file.shape[0]))




## === cell 11
train_file["language"].value_counts()




## === cell 12
language_name = train_file["language"].value_counts().index.tolist()
language_val = train_file["language"].value_counts().tolist()

fig = px.bar(
    x=language_name,
    y=language_val,
    title="Training Samples by Language",
    labels={"x": "Language", "y": "Sample count"},
    color=language_val,
)
fig.show()




## === cell 13
fig = px.pie(
    names=language_name,
    values=language_val,
    title="Training Samples by Language - Pie Chart",
    color_discrete_sequence=px.colors.sequential.RdBu_r,
)
fig.show()




## === cell 14
hindi = train_file[train_file["language"] == "hindi"]["context"].str.len()
tamil = train_file[train_file["language"] == "tamil"]["context"].str.len()

fig = make_subplots(rows=1, cols=2)

fig.add_trace(go.Histogram(x=list(hindi), name="Hindi Context"), row=1, col=1)
fig.add_trace(go.Histogram(x=list(tamil), name="Tamil Context"), row=1, col=2)

fig.update_layout(height=400, width=800, title_text="Character Count by Language")
iplot(fig)




## === cell 15
hindi = (
    train_file[train_file["language"] == "hindi"]["context"]
    .str.split()
    .map(lambda x: len(x))
)
tamil = (
    train_file[train_file["language"] == "tamil"]["context"]
    .str.split()
    .map(lambda x: len(x))
)

fig = make_subplots(rows=1, cols=2)

fig.add_trace(go.Histogram(x=list(hindi), name="Hindi Context"), row=1, col=1)
fig.add_trace(go.Histogram(x=list(tamil), name="Tamil Context"), row=1, col=2)

fig.update_layout(
    height=400, width=800, title_text="Word Count Distribution by Language"
)
iplot(fig)




## === cell 16
hindi = (
    train_file[train_file["language"] == "hindi"]["context"]
    .str.split()
    .map(lambda x: [len(j) for j in x])
    .map(lambda x: np.mean(x))
    .to_list()
)
tamil = (
    train_file[train_file["language"] == "tamil"]["context"]
    .str.split()
    .map(lambda x: [len(j) for j in x])
    .map(lambda x: np.mean(x))
    .to_list()
)

fig = ff.create_distplot([hindi, tamil], ["Hindi", "Tamil"])
fig.update_layout(
    height=500, width=800, title_text="Average Word Length Distribution by Language"
)
iplot(fig)




## === cell 17
hindi = (
    train_file[train_file["language"] == "hindi"]["context"]
    .apply(lambda x: len(set(str(x).split())))
    .to_list()
)
tamil = (
    train_file[train_file["language"] == "tamil"]["context"]
    .apply(lambda x: len(set(str(x).split())))
    .to_list()
)

fig = ff.create_distplot([hindi, tamil], ["Hindi", "Tamil"])
fig.update_layout(
    height=500, width=800, title_text="Unique Word Count Distribution by Language"
)
iplot(fig)




## === cell 18
question_to_answer = (
    train_file.groupby(train_file["question"].str.lower())["answer_text"]
    .apply(lambda x: x.iloc[0])
    .to_dict()
)

all_q_ans = list(
    zip(
        train_file["question"].astype(str).str.lower(),
        train_file["answer_text"].astype(str),
    )
)

prefix_len = 5
prefix_index = {}
for q, ans in zip(
    train_file["question"].astype(str), train_file["answer_text"].astype(str)
):
    q_low = q.lower()
    pref = q_low[:prefix_len]
    prefix_index.setdefault(pref, []).append((q_low, ans))


def best_fuzzy_match_prefix(q_low):
    """Return answer if a close fuzzy match (ratio ≥ 0.7) is found via prefix index."""
    pref = q_low[:prefix_len]
    candidates = prefix_index.get(pref, [])
    best_ratio = 0.0
    best_ans = None
    for cand_q, cand_ans in candidates:
        ratio = difflib.SequenceMatcher(None, q_low, cand_q).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_ans = cand_ans
    if best_ratio >= 0.7:  # lowered from 0.8
        return best_ans
    return None


def best_fuzzy_match_full(q_low, threshold=0.35):
    """Exhaustive fuzzy match against all training questions with a lower threshold."""
    best_ratio = 0.0
    best_ans = None
    for cand_q, cand_ans in all_q_ans:
        ratio = difflib.SequenceMatcher(None, q_low, cand_q).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_ans = cand_ans
    if best_ratio >= threshold:
        return best_ans
    return None


def select_sentence_by_overlap(context, question):
    """
    Split the context into sentences and return the one that shares the most
    words with the question. If no overlap is found, fall back to the first
    sentence. Truncate overly long sentences to keep the prediction concise.
    """
    split_chars = [".", "।", "!", "؟", "\n"]
    sentences = [context]
    for delim in split_chars:
        if delim in context:
            sentences = [s.strip() for s in context.split(delim) if s.strip()]
            break

    q_set = set(str(question).lower().split())
    best_sentence = None
    best_overlap = -1
    for sent in sentences:
        s_set = set(sent.lower().split())
        overlap = len(q_set & s_set)
        if overlap > best_overlap:
            best_overlap = overlap
            best_sentence = sent

    if best_sentence and best_overlap > 0:
        chosen = best_sentence
    elif sentences:
        chosen = sentences[0]
    else:
        chosen = context[:30].strip()

    return chosen[:30].strip()


predictions = []
for _, row in test_file.iterrows():
    q_lower = str(row["question"]).lower()
    if q_lower in question_to_answer:
        predictions.append(question_to_answer[q_lower])
        continue
    tfidf_ans = get_answer_by_tfidf(q_lower, min_score=0.15)
    if tfidf_ans is not None:
        predictions.append(tfidf_ans)
        continue
    fuzzy_ans = best_fuzzy_match_prefix(q_lower)
    if fuzzy_ans is not None:
        predictions.append(fuzzy_ans)
        continue
    fuzzy_ans = best_fuzzy_match_full(q_lower, threshold=0.35)
    if fuzzy_ans is not None:
        predictions.append(fuzzy_ans)
        continue
    ctx = str(row["context"])
    fallback = select_sentence_by_overlap(ctx, row["question"])
    predictions.append(fallback)

cprint(
    f"[green]Generated {len(predictions)} predictions using enhanced rule‑based fallback.[/green]"
)




## === cell 19
submission = pd.DataFrame()
submission["id"] = test_file["id"]
submission["PredictionString"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()




## === cell 20
cprint("[red]Under Work! More stuff coming soon[/red] ⚠")
