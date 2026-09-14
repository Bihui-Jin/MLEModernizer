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

# 5. Target score

0.3464473485946655

# 6. Current score

0.08125

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.04908) has done: 'I fix the import error by guarding the pipeline creation – if loading the local XLM‑RoBERTa model fails (protobuf incompatibility), the code fall back to a simple heuristic that returns the first few words of the context as a prediction. This ensures the `predictions` list is always defined and a valid `submission.csv` is written, allowing the notebook to run end‑to‑end and produce a submission file.'
- What this solution (achieved 0.075) has done: 'I fix the pipeline creation error by setting the device to CPU (‑1) and catching any exception. When the pipeline cannot be used, I replace the naive “first‑words” heuristic with a TF‑IDF nearest‑neighbor fallback that searches the training contexts for the most similar one and returns its known answer; this provides much more relevant predictions and should raise the Jaccard score toward the target. The rest of the logic (submission file creation) remains unchanged.'
- What this solution (achieved 0.01228) has done: 'I improve the fallback by (1) trying a reliable public QA model first, and (2) making the TF‑IDF similarity use both context + question, which gives much more relevant answers when the transformer model cannot be loaded. This keeps the original pipeline logic unchanged while boosting the Jaccard score toward the target.'
- What this solution (achieved 0.01228) has done: 'I guard the transformers import and pipeline creation inside a try/except so that protobuf import errors are caught and the TF‑IDF fallback runs without interruption. I also enhance the TF‑IDF fallback by returning the most common answer for the detected language when similarity is low, giving more sensible predictions and nudging the Jaccard score toward the target. The rest of the workflow stays unchanged, and a valid `submission.csv` is written.'
- What this solution (achieved 0.01228) has done: 'Implemented robust TF‑IDF fallback and improved fallback heuristics:

- **Cell 2**: TF‑IDF vectorizer now uses `min_df=1` to keep rare terms and adds `sublinear_tf=True` for better weighting.
- **Cell 3**: Enhanced fallback:
  * Uses the TF‑IDF similarity as before.
  * If the best similarity is low (≤ 0.1) or zero, returns the first few words of the context (a safe generic answer) instead of an empty string.
  * Guarantees a non‑empty prediction by also falling back to the most common answer per language when needed.
- Added explanatory comments for clarity.

These changes fix the pipeline loading error, ensure a non‑empty prediction for every test row, and improve answer relevance, moving the Jaccard score toward the target.'
- What this solution (achieved 0.01228) has done: 'Implemented a more reliable fallback for when the transformer QA pipeline cannot be loaded.  
The updated logic always returns the answer from the most similar training example using TF‑IDF similarity; if no similarity is found it falls back to the most common answer for the given language. This guarantees non‑empty, language‑aware predictions and improves the Jaccard score toward the target while keeping the original pipeline approach unchanged for environments where it works.'
- What this solution (achieved 0.01228) has done: 'Implemented robust fallback using multilingual SBERT embeddings to retrieve the most semantically similar training example when the transformer QA pipeline cannot be loaded. Added safe path handling for data files, ensured the SentenceTransformer model loads gracefully, and fall back to the original TF‑IDF approach only if SBERT import fails. This boosts answer relevance and moves the Jaccard score closer to the target while preserving the original pipeline logic.'
- What this solution (achieved 0.08125) has done: 'I add the missing imports and data loading, correct the variable names, and ensure the fallback logic runs safely. Minor tweaks to the SBERT similarity threshold and a fallback to the most‑common answer per language give a modest score boost while keeping the core approach unchanged. The script now creates a valid `submission.csv` file.'

# 9. Code solution

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

use_pipeline = False  # we will not use the heavy transformer pipeline
qa_pl = None  # placeholder kept for compatibility




## === cell 1

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
    most_common = group["answer_text"].fillna("").mode()
    most_common_per_lang[lang] = most_common.iloc[0] if not most_common.empty else "."

try:
    from sentence_transformers import SentenceTransformer

    sbert_model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    train_embeddings = sbert_model.encode(
        train_contexts,
        batch_size=256,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    use_sbert = True
    print("SBERT fallback model loaded successfully.")
except Exception as e:
    use_sbert = False
    print(f"Failed to load SBERT model ({e}); will rely on TF‑IDF only.")




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
predictions = []

for ctx, q, lang in test_df[["context", "question", "language"]].to_numpy():
    answer = None

    if use_sbert:
        test_vec = sbert_model.encode(
            [ctx + " " + q],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        sims = np.dot(
            train_embeddings, test_vec
        )  # cosine because embeddings are normalized
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]
        if best_sim > 0.2:
            answer = train_answers[best_idx]

    if answer is None:
        test_vec = tfidf_vectorizer.transform([ctx + " " + q])
        sims = linear_kernel(test_vec, tfidf_matrix).flatten()
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]
        if best_sim > 0.0:
            answer = train_answers[best_idx]
        else:
            answer = most_common_per_lang.get(lang, ".")

    if not answer or not isinstance(answer, str) or not answer.strip():
        answer = most_common_per_lang.get(lang, ".")

    predictions.append(answer)




## === cell 3
submission_df = pd.DataFrame(
    {"id": test_df["id"].values, "PredictionString": predictions}
)
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
submission_df.head()
