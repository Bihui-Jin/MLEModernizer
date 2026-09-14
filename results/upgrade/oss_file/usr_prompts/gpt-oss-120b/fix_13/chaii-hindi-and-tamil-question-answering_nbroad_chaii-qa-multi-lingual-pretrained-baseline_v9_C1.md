# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
transformers==4.53.3

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

# 5. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from pathlib import Path


def load_csv(name):
    possible_paths = [
        Path("/")
        / "kaggle"
        / "data"
        / "chaii-hindi-and-tamil-question-answering"
        / name,
        Path("/")
        / "kaggle"
        / "input"
        / "chaii-hindi-and-tamil-question-answering"
        / name,
        Path("/")
        / "kaggle"
        / "working"
        / "chaii-hindi-and-tamil-question-answering"
        / name,
        Path(name),
    ]
    for p in possible_paths:
        if p.is_file():
            return pd.read_csv(p)
    raise FileNotFoundError(f"{name} not found in any known location.")


train_df = load_csv("train.csv")
test_df = load_csv("test.csv")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

train_contexts = (
    train_df["context"].fillna("") + " " + train_df["question"].fillna("")
).tolist()
train_answers = train_df["answer_text"].fillna("").tolist()

tfidf_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True,
)
tfidf_matrix = tfidf_vectorizer.fit_transform(train_contexts)

most_common_per_lang = {}
for lang, group in train_df.groupby("language"):
    mode = group["answer_text"].fillna("").mode()
    most_common_per_lang[lang] = mode.iloc[0] if not mode.empty else "."

use_qa = False
qa_pipe = None
try:
    from transformers import pipeline

    qa_pipe = pipeline(
        "question-answering",
        model="deepset/xlm-roberta-large-squad2",
        tokenizer="deepset/xlm-roberta-large-squad2",
        device=-1,  # CPU
    )
    use_qa = True
    print("Transformer QA pipeline loaded (will be used when confident).")
except Exception as e:
    print(f"Failed to load QA pipeline ({e}); will rely on TF‑IDF / SBERT.")

use_sbert = False
sbert_model = None
train_embeddings = None
try:
    from sentence_transformers import SentenceTransformer

    sbert_model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    train_embeddings = sbert_model.encode(
        train_contexts,
        batch_size=256,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    use_sbert = True
    print("SBERT model loaded for fallback similarity.")
except Exception as e:
    print(f"Failed to load SBERT model ({e}); will skip SBERT fallback.")




## === cell 1
predictions = []

for row in test_df.itertuples(index=False, name=None):
    ctx, q, lang = row[1], row[2], row[3]
    answer = None

    if use_qa and qa_pipe is not None:
        try:
            result = qa_pipe(question=q, context=ctx)
            cand = result.get("answer", "").strip()
            score = result.get("score", 0.0)
            if cand and score >= 0.1:
                answer = cand
        except Exception as e:
            print(f"QA pipeline error for a row: {e}")

    if answer is None:
        test_vec = tfidf_vectorizer.transform([ctx + " " + q])
        sims = linear_kernel(test_vec, tfidf_matrix).flatten()
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]
        if best_sim > 0.0:
            answer = train_answers[best_idx]
        else:
            answer = None  # will trigger next fallback

    if answer is None and use_sbert and sbert_model is not None:
        try:
            test_emb = sbert_model.encode(
                [ctx + " " + q], normalize_embeddings=True, show_progress_bar=False
            )[0]
            sims = np.dot(train_embeddings, test_emb)
            best_idx = int(np.argmax(sims))
            best_sim = sims[best_idx]
            if best_sim > 0.3:
                answer = train_answers[best_idx]
        except Exception as e:
            print(f"SBERT fallback error: {e}")

    if not answer or not isinstance(answer, str) or not answer.strip():
        answer = most_common_per_lang.get(lang, ".")

    predictions.append(answer)

assert len(predictions) == len(test_df), "Prediction length mismatch!"




## === cell 2
submission_df = pd.DataFrame(
    {"id": test_df["id"].values, "PredictionString": predictions}
)
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
submission_df.head()
