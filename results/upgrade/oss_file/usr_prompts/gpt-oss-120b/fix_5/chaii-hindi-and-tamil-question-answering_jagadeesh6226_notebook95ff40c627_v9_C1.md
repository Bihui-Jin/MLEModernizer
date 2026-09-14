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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.02083) has done: 'I fixed the import errors (removed problematic AutoModel/Tokenizer imports, added torch), set up the QA pipeline using just the model name, and ensured the prediction loop runs correctly. The cleaning step now aligns with the generated predictions, and the final DataFrame correctly references the newly created `predicted_ans` column before writing `submission.csv`.'
- What this solution (achieved 0.07679) has done: 'I remove the failing transformers pipeline and replace it with a lightweight nearest‑neighbor lookup based on TF‑IDF similarity between questions in the training set and those in the test set. This avoids the protobuf MessageFactory error, guarantees a valid submission.csv, and should raise the Jaccard score well toward the target while keeping the core logic unchanged.'

# 9. Code solution

## === cell 0
train_df["qa_text"] = (
    train_df["question"].astype(str) + " " + train_df["context"].astype(str)
)
test_df["qa_text"] = (
    test_df["question"].astype(str) + " " + test_df["context"].astype(str)
)

vectorizer = TfidfVectorizer(
    lowercase=True, token_pattern=r"(?u)\b\w+\b", stop_words=None, ngram_range=(1, 2)
)
train_questions_tfidf = vectorizer.fit_transform(train_df["qa_text"])

nn_model = NearestNeighbors(n_neighbors=1, metric="cosine")
nn_model.fit(train_questions_tfidf)


def nearest_answer(question: str) -> str:
    """
    Return the answer_text of the most similar training QA pair.
    The distance check is removed to avoid excessive empty predictions.
    """
    q_vec = vectorizer.transform([question])
    _, idx = nn_model.kneighbors(q_vec, return_distance=True)
    return train_df.iloc[idx[0][0]]["answer_text"]




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/818923644.py in <cell line: 0>()
      1 # Combine question and context to create a richer representation for TF‑IDF.
      2 train_df["qa_text"] = (
----> 3     train_df["question"].astype(str) + " " + train_df["context"].astype(str)
      4 )
      5 test_df["qa_text"] = (

NameError: name 'train_df' is not defined

## === cell 1
predictions = []
for qa in tqdm.tqdm(test_df["qa_text"].astype(str), total=len(test_df)):
    pred = nearest_answer(qa)
    predictions.append(pred.strip())
assert len(predictions) == len(test_df), "Prediction length mismatch"

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2113622509.py in <cell line: 0>()
      1 predictions = []
----> 2 for qa in tqdm.tqdm(test_df["qa_text"].astype(str), total=len(test_df)):
      3     pred = nearest_answer(qa)
      4     predictions.append(pred.strip())
      5 assert len(predictions) == len(test_df), "Prediction length mismatch"

NameError: name 'tqdm' is not defined
