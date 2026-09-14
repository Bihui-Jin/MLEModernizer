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

0.4563716948032379

# 6. Current score

0.03027

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00943) has done: 'I fixed the crash caused by the unavailable XLM‑RoBERTa model by wrapping the pipeline creation in a try/except block. If the model can’t be loaded, the code now falls back to a lightweight heuristic that extracts the first sentence from the context as the answer. This guarantees that a `predictions` list is always created, allowing the submission CSV to be written without errors.'
- What this solution (achieved 0.01686) has done: 'The changes wrap the `transformers` import in a try/except so the script no longer crashes when protobuf incompatibilities arise, and replace the simple first‑sentence fallback with a slightly smarter heuristic that selects the sentence in the context having the most word overlap with the question. This keeps the original pipeline logic when the model loads, while providing a deterministic fallback that should improve the Jaccard score toward the target without altering the core model architecture. The script now always produces a valid `submission.csv` file.'
- What this solution (achieved 0.03027) has done: 'I remove the failing transformer pipeline and make the script always use a robust fallback heuristic. The heuristic is enhanced to first look for any numeric token in the context (common in this dataset) and, if none are found, fall back to picking the sentence with the most word overlap with the question. This avoids the protobuf error, guarantees a valid `submission.csv`, and modestly improves the Jaccard score while preserving the overall workflow.'
- What this solution (achieved 0.03027) has done: 'I load a lightweight multilingual sentence‑transformer once and use it inside the fallback `heuristic_answer`. If the model loads successfully we embed the question and each sentence of the context and pick the sentence with the highest cosine similarity; otherwise we keep the original word‑overlap heuristic (with the numeric token shortcut). This adds a more semantic matching step, which should raise the Jaccard score toward the target while still guaranteeing a valid `submission.csv`.'
- What this solution (achieved 0.03027) has done: 'Implemented robust imports and added a TF‑IDF fallback for sentence similarity. The code now safely handles missing `sentence_transformers`, uses TF‑IDF cosine similarity when embeddings aren’t available, and otherwise falls back to the original word‑overlap heuristic. This improves answer selection without changing the core pipeline and guarantees a valid `submission.csv`.'
- What this solution (achieved 0.03027) has done: 'The fix moves the protobuf implementation setting before any imports, safely disables the problematic SentenceTransformer import, and improves the fallback heuristic by preprocessing text for TF‑IDF similarity (lowercasing and removing punctuation). This prevents the AttributeError, guarantees a valid `submission.csv`, and modestly raises the Jaccard score toward the target while keeping the core logic unchanged.'
- What this solution (achieved 0.03027) has done: 'Implemented robust fallback handling by disabling the problematic SentenceTransformer import and adding a lightweight multilingual QA pipeline from 🤗 transformers. The script now safely attempts to load the QA model; if unavailable it falls back to TF‑IDF or simple word‑overlap heuristics. This prevents the protobuf‑related crash, ensures a valid `submission.csv`, and provides a stronger answer extraction method likely to raise the Jaccard score toward the target.'
- What this solution (achieved 0.03027) has done: 'The fix removes the risky import of the transformers QA pipeline, which caused an uncaught protobuf‑related AttributeError. By safely disabling the pipeline and keeping the fallback heuristic (numeric token → TF‑IDF → word overlap), the script runs end‑to‑end, always creates a valid `submission.csv`, and retains the same core logic while modestly improving the Jaccard score.'
- What this solution (achieved 0.03027) has done: 'I add safe loading of a lightweight multilingual QA pipeline and a sentence‑transformer model, keeping all existing fall‑backs. This lets the script use a proper QA model when available (raising the Jaccard score toward the target) while still guaranteeing a valid submission.csv through the original heuristics if the models fail to load.'
- What this solution (achieved 0.03027) has done: 'The fix removes the fragile imports of the `sentence_transformers` and `transformers` QA pipeline, which raise an uncaught protobuf `AttributeError`. We keep only the safe TF‑IDF utilities (which are available) and set the corresponding flags to `False`. The fallback heuristic now relies on numeric tokens, TF‑IDF similarity, and word‑overlap, guaranteeing that the script runs end‑to‑end and writes a valid `submission.csv` while staying within the original logic.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import re
import pandas as pd
import torch
import torch.nn.functional as F

SENTENCE_TRANSFORMERS_AVAILABLE = False
EMBED_MODEL = None
QA_PIPELINE = None
QA_AVAILABLE = False
TFIDF_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    TFIDF_AVAILABLE = True
except Exception as e:
    print("scikit‑learn TF‑IDF utilities not available:", e)


test_path = "/kaggle/input/chaii-hindi-and-tamil-question-answering/test.csv"
test_df = pd.read_csv(test_path)




## === cell 1
def _preprocess(text: str) -> str:
    """Lower‑case and strip punctuation for more robust TF‑IDF matching."""
    return re.sub(r"[^\w\s]", " ", text.lower())


def heuristic_answer(context: str, question: str) -> str:
    """
    Fallback answer generator with multiple strategies:
    1. Return the first numeric token found in the context.
    2. If a QA pipeline is available, use it.   (not used here)
    3. If sentence‑transformer embeddings are available, select the most similar sentence.
    4. Else if TF‑IDF is available, select the sentence with highest TF‑IDF similarity.
    5. Otherwise, pick the sentence with the greatest word‑overlap with the question.
    """
    numeric_match = re.search(r"\b\d+\b", context)
    if numeric_match:
        return numeric_match.group(0)

    sentences = re.split(r"[\.!\?]\s+", context.strip())
    if not sentences:
        return ""



    if TFIDF_AVAILABLE:
        try:
            processed_q = _preprocess(question)
            processed_sents = [_preprocess(s) for s in sentences]
            vectorizer = TfidfVectorizer().fit([processed_q] + processed_sents)
            q_vec = vectorizer.transform([processed_q])
            s_vec = vectorizer.transform(processed_sents)
            cos_sims = cosine_similarity(q_vec, s_vec).flatten()
            best_idx = int(cos_sims.argmax())
            return sentences[best_idx]
        except Exception:
            pass  # fallback to word‑overlap if TF‑IDF fails

    q_tokens = set(question.lower().split())
    best_sentence = sentences[0]
    best_overlap = -1
    for sent in sentences:
        s_tokens = set(sent.lower().split())
        overlap = len(q_tokens & s_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_sentence = sent
    return best_sentence




## === cell 2
predictions = []
for row in test_df.itertuples(index=False):
    ctx = row.context
    q = row.question
    predictions.append(heuristic_answer(ctx, q))




## === cell 3
submission_df = pd.DataFrame({"id": test_df["id"], "PredictionString": predictions})
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
