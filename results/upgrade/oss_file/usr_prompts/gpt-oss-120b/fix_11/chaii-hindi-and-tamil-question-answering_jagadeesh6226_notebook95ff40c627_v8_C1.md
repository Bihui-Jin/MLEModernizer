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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
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

0.467526763677597

# 6. Current score

0.08646

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the failing imports, load a publicly‑available QA model (roberta‑base‑squad2) that works with the current Transformers version, adjust the prediction function to handle long inputs safely, make sure the prediction list length matches the test set, clean the text, and finally write a proper `submission.csv` with the required `PredictionString` column.'
- What this solution (achieved 0.0) has done: 'I replace the failing Transformers model loading with a simple lookup‑based predictor that reuses the training answers: if a test question appears in the training set we return its most common answer, otherwise we return an empty string. This removes the protobuf error, ensures a valid `submission.csv` is written, and gives a non‑zero Jaccard score that moves the result toward the target. The rest of the pipeline (cleaning, CSV creation) remains unchanged.'
- What this solution (achieved 0.08125) has done: 'I add a lightweight semantic‑matching step using a multilingual SentenceTransformer: first try the existing exact‑question lookup, and if it fails I retrieve the most similar training context (by cosine similarity of normalized embeddings) and use its answer. This keeps the original logic intact while providing many more non‑empty predictions, which should raise the Jaccard score toward the target. The rest of the pipeline (cleaning, CSV creation) remains unchanged.'
- What this solution (achieved 0.08125) has done: 'I fixed the import error by making the SentenceTransformer import optional and added a fallback TF‑IDF based embedding pipeline (using scikit‑learn) that works without protobuf. The rest of the logic stays the same, so the script now runs end‑to‑end, creates a valid `submission.csv`, and the simple semantic matching should improve the Jaccard score toward the target.'
- What this solution (achieved 0.08021) has done: 'I removed the problematic sentence_transformers import and replaced it with a direct transformers + torch embedding pipeline that works with the current library versions. The new code builds multilingual MiniLM embeddings for both training and test contexts, normalizes them, and then uses cosine similarity to retrieve the most similar training answer when an exact question match isn’t found. This fixes the import error, ensures a valid submission.csv is written, and provides a stronger semantic‑matching step to raise the Jaccard score toward the target.'
- What this solution (achieved 0.075) has done: 'I disable the failing MiniLM SBERT embedding by setting `_use_sbert = False`, which avoids the protobuf‑related error. Then I keep the TF‑IDF fallback for encoding contexts. In the prediction loop I make the similarity computation robust to both dense NumPy arrays and sparse SciPy matrices by converting sparse results to a dense vector before taking the argmax. These minimal changes eliminate the runtime crash and ensure a proper `submission.csv` is written, while preserving the existing logic that already improves the Jaccard score.'
- What this solution (achieved 0.05665) has done: 'I improve the fallback prediction by matching on question similarity instead of context similarity, which is more directly related to the answer. The script now builds a separate TF‑IDF embedding for questions, uses a cleaned version of the test question for exact‑match lookup, and retrieves the most similar training answer via question embeddings. This modest change is expected to raise the Jaccard score toward the target while preserving the original pipeline.'
- What this solution (achieved 0.08522) has done: 'The changes lower‑case question keys for a more robust exact‑match lookup and, when no exact match is found, compare both question‑ and context‑based cosine similarities (using the existing TF‑IDF vectors). The answer from the side with the higher similarity (and above a modest threshold) is chosen, which adds useful semantic information while keeping the original pipeline intact and moving the Jaccard score nearer to the target.'
- What this solution (achieved 0.08522) has done: 'I add a fallback to the most frequent non‑empty answer and lower the similarity threshold, so the model predicts a non‑empty string for more test rows. This small change keeps the original exact‑match and TF‑IDF similarity logic while increasing recall, which should raise the Jaccard score toward the target.'
- What this solution (achieved 0.08646) has done: 'I add a lightweight text‑normalisation for questions, combine question‑ and context‑based TF‑IDF similarities with weighted averaging, and raise the similarity threshold slightly. This keeps the original TF‑IDF pipeline but makes matching more robust and selective, which should raise the Jaccard score toward the target while preserving the core logic.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import re
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel

_use_sbert = False



## === cell 1
test_df = pd.read_csv("../input/chaii-hindi-and-tamil-question-answering/test.csv")



## === cell 2
train_df = pd.read_csv("../input/chaii-hindi-and-tamil-question-answering/train.csv")


def _normalize(txt: str) -> str:
    txt = txt.lower().strip()
    txt = re.sub(r"[^\w\s]", " ", txt)  # remove punctuation
    txt = re.sub(r"\s+", " ", txt)  # collapse multiple spaces
    return txt


question_to_answer = (
    train_df.groupby("question")["answer_text"]
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "")
    .to_dict()
)
question_to_answer = {_normalize(q): ans for q, ans in question_to_answer.items()}

train_answers = train_df["answer_text"].astype(str).tolist()

non_empty_answers = train_df.loc[
    train_df["answer_text"].astype(str) != "", "answer_text"
]
if not non_empty_answers.empty:
    most_common_answer = non_empty_answers.mode().iloc[0]
else:
    most_common_answer = ""  # safety fallback



## === cell 3
if _use_sbert:
    print("Encoding training and test contexts with MiniLM...")
    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    model = AutoModel.from_pretrained(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    model.eval()
    model.to("cpu")  # use CPU (GPU not guaranteed in Kaggle environment)

    def encode_texts(texts, batch_size=256):
        all_embeds = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                encoded = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                encoded = {k: v.to("cpu") for k, v in encoded.items()}
                outputs = model(**encoded)
                last_hidden = outputs.last_hidden_state  # (bs, seq_len, dim)
                attention = encoded["attention_mask"].unsqueeze(-1)  # (bs, seq_len, 1)
                masked = last_hidden * attention
                embeddings = masked.sum(dim=1) / attention.sum(dim=1)  # mean pooling
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeds.append(embeddings.cpu().numpy())
        return np.vstack(all_embeds)

    train_contexts = train_df["context"].astype(str).tolist()
    train_embeddings = encode_texts(train_contexts)

    test_contexts = test_df["context"].astype(str).tolist()
    test_embeddings = encode_texts(test_contexts)
else:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    print("Encoding training contexts with TF‑IDF fallback...")
    vectorizer_ctx = TfidfVectorizer(
        sublinear_tf=True, max_features=50000, stop_words=None, ngram_range=(1, 2)
    )
    train_contexts = train_df["context"].astype(str).tolist()
    train_embeddings = vectorizer_ctx.fit_transform(train_contexts)
    train_embeddings = normalize(train_embeddings)

    print("Encoding test contexts with TF‑IDF fallback...")
    test_contexts = test_df["context"].astype(str).tolist()
    test_embeddings = vectorizer_ctx.transform(test_contexts)
    test_embeddings = normalize(test_embeddings)

    print("Encoding training and test questions with TF‑IDF...")
    vectorizer_q = TfidfVectorizer(
        sublinear_tf=True, max_features=50000, stop_words=None, ngram_range=(1, 2)
    )
    train_questions = train_df["question"].astype(str).tolist()
    train_q_embeddings = vectorizer_q.fit_transform(train_questions)
    train_q_embeddings = normalize(train_q_embeddings)

    test_questions = test_df["question"].astype(str).tolist()
    test_q_embeddings = vectorizer_q.transform(test_questions)
    test_q_embeddings = normalize(test_q_embeddings)



## === cell 4
question_lst = test_df["question"].tolist()

predict_list = []
SIM_THRESHOLD = 0.10  # a modest raise to keep only reasonably similar matches
WEIGHT_Q = 0.7
WEIGHT_CTX = 0.3

for idx, q in enumerate(tqdm(question_lst, total=len(question_lst))):
    q_clean = _normalize(q)
    pred = question_to_answer.get(q_clean, "")
    if not pred:
        sims_q = train_q_embeddings @ test_q_embeddings[idx].T
        if hasattr(sims_q, "toarray"):
            sims_q = sims_q.toarray().ravel()
        else:
            sims_q = np.squeeze(sims_q)

        sims_ctx = train_embeddings @ test_embeddings[idx].T
        if hasattr(sims_ctx, "toarray"):
            sims_ctx = sims_ctx.toarray().ravel()
        else:
            sims_ctx = np.squeeze(sims_ctx)

        combined_sims = WEIGHT_Q * sims_q + WEIGHT_CTX * sims_ctx
        best_idx = int(np.argmax(combined_sims))
        best_score = combined_sims[best_idx]

        if best_score >= SIM_THRESHOLD:
            pred = train_answers[best_idx]
        else:
            pred = most_common_answer
    predict_list.append(pred)



## === cell 5
test_df["predicted_ans"] = predict_list



## === cell 6
reg_patterns = [
    r"^[0-9]+\.",  # 1., 12.
    r"^[0-9]\.",  # 1.
    r"^[0-9][0-9]\.",  # 12.
    r"^[0-9][0-9][0-9]\.",  # 123.
    r"^\([0-9]+\)",  # (1)
    r"^[0-9]+\)",  # 1)
]


def clean_text(txt):
    for pat in reg_patterns:
        txt = re.sub(pat, " ", txt)
    txt = txt.replace("(", " ").replace(")", " ")
    return txt.strip()


test_df["predicted_ans"] = test_df["predicted_ans"].apply(clean_text)



## === cell 7
submission = pd.DataFrame(
    {"id": test_df["id"], "PredictionString": test_df["predicted_ans"]}
)



## === cell 8
submission.to_csv("submission.csv", index=False)
