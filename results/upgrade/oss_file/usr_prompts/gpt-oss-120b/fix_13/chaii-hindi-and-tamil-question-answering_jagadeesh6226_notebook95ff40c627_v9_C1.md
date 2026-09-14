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

3.10

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tqdm==4.67.1
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

0.5884965658187866

# 6. Current score

0.00116

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.02083) has done: 'I fixed the import errors (removed problematic AutoModel/Tokenizer imports, added torch), set up the QA pipeline using just the model name, and ensured the prediction loop runs correctly. The cleaning step now aligns with the generated predictions, and the final DataFrame correctly references the newly created `predicted_ans` column before writing `submission.csv`.'
- What this solution (achieved 0.07679) has done: 'I remove the failing transformers pipeline and replace it with a lightweight nearest‑neighbor lookup based on TF‑IDF similarity between questions in the training set and those in the test set. This avoids the protobuf MessageFactory error, guarantees a valid submission.csv, and should raise the Jaccard score well toward the target while keeping the core logic unchanged.'
- What this solution (achieved 0.08393) has done: 'I correct the data file paths so the script can locate the training and test CSVs in the Kaggle environment (checking common directories). This fixes the FileNotFoundError and ensures all subsequent cells have access to `train_df` and `test_df`, allowing the TF‑IDF nearest‑neighbor prediction loop to run and produce a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.06652) has done: 'I keep the overall TF‑IDF + nearest‑neighbor pipeline but make two small tweaks that are likely to raise the Jaccard score: (1) retrieve the 5 closest training examples instead of only the single nearest one, and (2) pick the candidate answer that shares the most words with the test question (using the provided Jaccard function). This stays within the original logic while giving a modest performance boost toward the target score.'
- What this solution (achieved 0.02277) has done: 'Implemented language‑aware nearest‑neighbor lookup and expanded the neighbor pool to improve relevance of the retrieved answer. The `nn_model` now considers up to 15 neighbours, and `nearest_answer` filters candidates by matching language and scores answers using Jaccard similarity with the original **question** (instead of the combined text). The prediction loop passes both the combined QA text and language to the function, preserving the original pipeline while modestly boosting the Jaccard score toward the target.'
- What this solution (achieved 0.01915) has done: 'Implemented three lightweight enhancements while keeping the original TF‑IDF + nearest‑neighbor pipeline intact:  
1. Raised the neighbour pool to 50 to give richer candidate answers.  
2. Modified `nearest_answer` to score candidates using Jaccard similarity between the candidate answer and the **test question** (a more relevant proxy) and added a fallback to the non‑language‑filtered neighbours when no matching language is found.  
3. Updated the prediction loop to pass the test question into the revised function. These adjustments are expected to raise the Jaccard‑based score toward the target without altering the core modelling approach.'
- What this solution (achieved 0.07202) has done: 'Implemented a focused tweak in the nearest‑neighbor lookup: instead of scoring candidates by Jaccard similarity to the test question, the function now directly returns the answer from the closest language‑matched neighbour (or the overall closest if none match). This aligns the prediction with the most similar training QA pair, which should raise the Jaccard score toward the target while preserving the existing TF‑IDF + NearestNeighbors pipeline.'
- What this solution (achieved 0.02361) has done: 'Implemented a language‑aware nearest‑neighbor lookup that scores candidate answers by Jaccard similarity to the test question and refines the neighbor ranking with a combined TF‑IDF distance (using both full QA text and question‑only vectors). This modest enhancement stays within the original TF‑IDF + NearestNeighbors pipeline while aiming to move the Jaccard score closer to the target.'
- What this solution (achieved 0.00116) has done: 'Implemented a modest yet targeted enhancement to the TF‑IDF nearest‑neighbor pipeline:

- Expanded the neighbor pool to 200 to give a richer set of candidate answers.
- Adjusted `nearest_answer` to also receive the test `context` and score candidates by Jaccard similarity against this context (the answer is expected to be a text snippet from the context). This more directly reflects the evaluation metric.
- Kept the original language‑aware filtering and fallback logic intact, preserving the core TF‑IDF + NearestNeighbors approach while nudging the Jaccard score closer to the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity


def locate_file(relative_path: str) -> str:
    """
    Search common Kaggle directories for the given relative file path.
    Returns the first existing full path or raises FileNotFoundError.
    """
    search_roots = [
        "./data",
        "./input",
        "/kaggle/input",
        "./kaggle/input",
        "data",
        "input",
    ]
    for root in search_roots:
        candidate = os.path.join(root, relative_path)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find {relative_path} in any known location.")


train_path = locate_file("chaii-hindi-and-tamil-question-answering/train.csv")
test_path = locate_file("chaii-hindi-and-tamil-question-answering/test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)




## === cell 1
train_df["qa_text"] = (
    train_df["question"].astype(str) + " " + train_df["context"].astype(str)
)
test_df["qa_text"] = (
    test_df["question"].astype(str) + " " + test_df["context"].astype(str)
)

vectorizer = TfidfVectorizer(
    lowercase=True,
    token_pattern=r"(?u)\b\w+\b",
    stop_words=None,
    ngram_range=(1, 2),
    sublinear_tf=True,
)

train_questions_tfidf = vectorizer.fit_transform(train_df["qa_text"])

vectorizer_q = TfidfVectorizer(
    lowercase=True,
    token_pattern=r"(?u)\b\w+\b",
    stop_words=None,
    ngram_range=(1, 2),
    sublinear_tf=True,
)

train_questions_only_tfidf = vectorizer_q.fit_transform(train_df["question"])

nn_model = NearestNeighbors(n_neighbors=200, metric="cosine")
nn_model.fit(train_questions_tfidf)




## === cell 2
def jaccard(str1: str, str2: str) -> float:
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return (
        float(len(c)) / (len(a) + len(b) - len(c))
        if (len(a) + len(b) - len(c)) > 0
        else 0.0
    )


def nearest_answer(qa_text: str, question: str, language: str, context: str) -> str:
    """
    Retrieve the most suitable answer from the training set.
    - Uses the primary TF‑IDF (QA) vector to fetch up to 200 neighbours.
    - Refines ranking with a secondary question‑only TF‑IDF distance.
    - Scores candidates by Jaccard similarity between the candidate answer
      and the test *context* (the answer is expected to be a snippet of the context).
    - Falls back to the first neighbour if no answer passes the checks.
    """
    q_vec = vectorizer.transform([qa_text])
    distances_qa, indices = nn_model.kneighbors(q_vec, return_distance=True)
    candidate_idxs = indices[0]
    distances_qa = distances_qa[0]

    q_only_vec = vectorizer_q.transform([question])
    sim_q = cosine_similarity(
        q_only_vec, train_questions_only_tfidf[candidate_idxs]
    ).flatten()
    distances_q = 1.0 - sim_q

    combined_dist = (distances_qa + distances_q) / 2.0

    lang_mask = train_df.iloc[candidate_idxs]["language"] == language
    filtered_idxs = np.array(candidate_idxs)[lang_mask.values]
    filtered_combined = combined_dist[lang_mask.values]

    if len(filtered_idxs) > 0:
        search_idxs = filtered_idxs
        search_dist = filtered_combined
    else:
        search_idxs = candidate_idxs
        search_dist = combined_dist

    best_answer = ""
    best_jacc = -1.0
    best_dist = float("inf")

    for idx, dist in zip(search_idxs, search_dist):
        ans = train_df.iloc[int(idx)]["answer_text"]
        if pd.isna(ans):
            continue
        ans_str = str(ans).strip()
        cur_jacc = jaccard(ans_str, context)
        if (cur_jacc > best_jacc) or (cur_jacc == best_jacc and dist < best_dist):
            best_jacc = cur_jacc
            best_dist = dist
            best_answer = ans_str

    if best_answer == "":
        first_idx = candidate_idxs[0]
        best_answer = str(train_df.iloc[int(first_idx)]["answer_text"]).strip()
    return best_answer




## === cell 3
predictions = []
for _, row in tqdm.tqdm(test_df.iterrows(), total=len(test_df)):
    pred = nearest_answer(
        row["qa_text"], row["question"], row["language"], row["context"]
    )
    predictions.append(pred.strip())
assert len(predictions) == len(test_df), "Prediction length mismatch"




## === cell 4
submission = pd.DataFrame({"id": test_df["id"], "PredictionString": predictions})

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}, shape: {submission.shape}")
