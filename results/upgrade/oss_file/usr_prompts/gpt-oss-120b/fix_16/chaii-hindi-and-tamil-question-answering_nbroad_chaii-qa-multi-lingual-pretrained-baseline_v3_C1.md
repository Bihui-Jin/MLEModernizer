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

0.5439859628677368

# 6. Current score

0.07351

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.07679) has done: 'The fix removes the failing HuggingFace pipeline (which crashes due to a protobuf incompatibility) and replaces it with a lightweight TF‑IDF similarity baseline: it vectorises the concatenated context + question from the training set, finds the most similar training example for each test row, and uses that example’s answer_text as the prediction. This restores end‑to‑end execution, creates a valid `submission.csv`, and provides a reasonable baseline that moves the Jaccard score toward the target.'
- What this solution (achieved 0.07351) has done: 'I extend the TF‑IDF representation to also include the ground‑truth answer text when building the training documents and broaden the n‑gram range with more features, so the similarity search can better match answer content. This small change keeps the same nearest‑neighbor‑based prediction logic while giving the model richer information, which should raise the Jaccard score toward the target.'
- What this solution (achieved 0.07619) has done: 'I split the training data by language and build separate TF‑IDF models for each language, then during inference use the model that matches the test row’s language. This reduces noise from unrelated language examples and usually yields higher similarity scores, moving the Jaccard score noticeably closer to the target. I also keep a global fallback model in case a language is missing, and I increase the n‑gram range to (1, 4) for richer coverage while preserving the original pipeline structure.'
- What this solution (achieved 0.07619) has done: 'I replace the raw TF‑IDF dot‑product similarity with cosine similarity, which better reflects true vector similarity and usually yields higher Jaccard scores while keeping the same nearest‑neighbor workflow and language‑specific models. If a language‑specific model returns only zero similarities, the code fall back to the global model to avoid empty predictions.'
- What this solution (achieved 0.07351) has done: 'I add character‑level TF‑IDF features (using `analyzer='char_wb'`) and combine them with the existing word‑level TF‑IDF vectors. The two sparse matrices are horizontally stacked, so similarity is computed on a richer representation while preserving the original nearest‑neighbor logic and language‑specific handling. This modest enrichment should increase the Jaccard score toward the target without changing the overall pipeline.'
- What this solution (achieved 0.06458) has done: 'I boost the influence of the ground‑truth answer text (by repeating it a few times) and slightly enlarge the TF‑IDF n‑gram windows, which should make the nearest‑neighbor search retrieve more relevant training rows and raise the Jaccard score toward the target, while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.08244) has done: 'We reduce the influence of the answer text in the TF‑IDF vectors, because the test data does not contain it. By setting `ANSWER_REPEAT` to 0 and building the training vectors from only *context + question* (the answer is still used as the final prediction), we keep the same nearest‑neighbor pipeline while making similarity matching more relevant, which is expected to raise the Jaccard score toward the target.'
- What this solution (achieved 0.08244) has done: 'I added a quick exact‑question lookup so that when a test question already appears in the training set we return the known answer directly, bypassing the TF‑IDF nearest‑neighbor search. This simple rule captures many perfect matches and should raise the Jaccard score toward the target while keeping the original pipeline unchanged. The rest of the code (TF‑IDF models, language‑specific handling, and CSV output) remains the same.'
- What this solution (achieved 0.07351) has done: 'I keep the overall TF‑IDF nearest‑neighbor pipeline but reintegrate the ground‑truth answer text into the training vectors (setting `ANSWER_REPEAT = 1`). Adding the answer once makes the vectors richer and improves similarity matching, which is expected to raise the Jaccard score toward the target while preserving the existing logic.'
- What this solution (achieved 0.08244) has done: 'I set `ANSWER_REPEAT` to 0 so training vectors no longer contain answer text, which makes similarity focus on context + question and has been shown to raise the Jaccard score. I also add a low‑similarity fallback: when the best cosine similarity for a language‑specific model is below a modest threshold (e.g., 0.2) the code fall back to the global model, avoiding poor matches that hurt the metric. These small adjustments keep the original TF‑IDF nearest‑neighbor pipeline intact while nudging the score toward the target.'
- What this solution (achieved 0.08244) has done: 'I keep the overall TF‑IDF nearest‑neighbor pipeline unchanged but improve the matching step. Instead of using a fixed similarity threshold that forces a fallback to the global model, I compute both the language‑specific and the global similarities for every test item and select the answer from the model (language‑specific or global) that yields the higher cosine similarity. This small change lets the system keep a more relevant language‑specific match when it is strong while still benefiting from the global model when it is better, which should raise the Jaccard score toward the target without altering the core logic.'
- What this solution (achieved 0.07798) has done: 'I replace the TF‑IDF similarity pipeline with a multilingual Sentence‑Transformer embedding model (distiluse‑base‑multilingual‑cased‑v2).  The code still keeps the exact‑question lookup and language‑specific handling, but now computes cosine similarity on dense embeddings, which aligns much better with the semantic matching needed for the Jaccard metric and should raise the score toward the target.  All other steps (data loading, CSV output) remain unchanged, and the submission file is still written as `submission.csv`.'
- What this solution (achieved 0.075) has done: 'Implemented a robust TF‑IDF based similarity pipeline that replaces the failing SentenceTransformer approach. The new code builds global and language‑specific TF‑IDF vectors (using 1‑4 word n‑grams), leverages cosine similarity for nearest‑neighbor matching, and retains the exact‑question lookup. This fixes the protobuf error, ensures a valid `submission.csv` is generated, and modestly boosts the Jaccard score toward the target while keeping the overall workflow unchanged.'
- What this solution (achieved 0.07083) has done: 'I enrich the TF‑IDF representation by adding character‑level n‑gram features and concatenating them with the existing word‑level vectors. Both the global model and each language‑specific model now keep a word and a character vectorizer, and the test vectors are built by stacking the two transformed sparse matrices before computing cosine similarity. This small augmentation keeps the original nearest‑neighbor logic unchanged while giving the similarity search more expressive features, which should raise the Jaccard score toward the target.'
- What this solution (achieved 0.07351) has done: 'I adjust the TF‑IDF pipeline to give it a richer representation that is still based on the original nearest‑neighbor logic.  
Key changes:  

* Enable `ANSWER_REPEAT = 1` so the answer text is included once in the training documents, providing more signal for similarity matching.  
* Use `sublinear_tf=True` and `norm='l2'` in both word‑level and character‑level TF‑IDF vectorizers to improve weighting of informative terms.  
* Expand the character n‑gram range to `(2, 5)` to capture finer patterns in Hindi/Tamil scripts.  

These tweaks keep the overall workflow unchanged (lookup, language‑specific models, cosine similarity) while modestly strengthening the similarity scoring, which should raise the Jaccard score toward the target.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

train_path = "../input/chaii-hindi-and-tamil-question-answering/train.csv"
test_path = "../input/chaii-hindi-and-tamil-question-answering/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

ANSWER_REPEAT = 1  # was 0


def make_combined_text(row):
    text = f"{row['context']} {row['question']}"
    if ANSWER_REPEAT > 0 and pd.notna(row.get("answer_text")):
        text += (" " + str(row["answer_text"])) * ANSWER_REPEAT
    return text


question_answer_lookup = {
    (row["question"].strip(), row["language"]): row["answer_text"]
    for _, row in train_df.iterrows()
    if pd.notna(row["answer_text"])
}


## === cell 1
global_texts = train_df.apply(make_combined_text, axis=1).astype(str).tolist()

vec_word_global = TfidfVectorizer(
    ngram_range=(1, 4), analyzer="word", sublinear_tf=True, norm="l2"
)
mat_word_global = vec_word_global.fit_transform(global_texts)

vec_char_global = TfidfVectorizer(
    ngram_range=(2, 5), analyzer="char_wb", sublinear_tf=True, norm="l2"
)
mat_char_global = vec_char_global.fit_transform(global_texts)

global_matrix = hstack([mat_word_global, mat_char_global]).tocsr()
global_answers = train_df["answer_text"].reset_index(drop=True)

lang_data = {}
for lang in train_df["language"].unique():
    mask = train_df["language"] == lang
    texts_lang = (
        train_df.loc[mask].apply(make_combined_text, axis=1).astype(str).tolist()
    )

    vec_word = TfidfVectorizer(
        ngram_range=(1, 4), analyzer="word", sublinear_tf=True, norm="l2"
    )
    mat_word = vec_word.fit_transform(texts_lang)

    vec_char = TfidfVectorizer(
        ngram_range=(2, 5), analyzer="char_wb", sublinear_tf=True, norm="l2"
    )
    mat_char = vec_char.fit_transform(texts_lang)

    mat_combined = hstack([mat_word, mat_char]).tocsr()

    lang_data[lang] = {
        "vec_word": vec_word,
        "vec_char": vec_char,
        "matrix": mat_combined,
        "answers": train_df.loc[mask, "answer_text"].reset_index(drop=True),
    }


## === cell 2
predictions = []


def get_best_match(test_vec, train_mat, answers):
    sims = cosine_similarity(train_mat, test_vec)[:, 0]  # (n_train,)
    best_idx = np.argmax(sims)
    return answers.iloc[best_idx], sims[best_idx]


for _, row in test_df.iterrows():
    test_question = row["question"]
    lang = row["language"]

    lookup_key = (test_question.strip(), lang)
    if lookup_key in question_answer_lookup:
        pred = question_answer_lookup[lookup_key]
        predictions.append(str(pred) if pd.notna(pred) else "")
        continue

    test_text = f"{row['context']} {row['question']}"

    test_vec_word_global = vec_word_global.transform([test_text])
    test_vec_char_global = vec_char_global.transform([test_text])
    test_vec_global = hstack([test_vec_word_global, test_vec_char_global])
    pred_global, best_sim_global = get_best_match(
        test_vec_global, global_matrix, global_answers
    )

    pred_lang, best_sim_lang = None, -1.0
    if lang in lang_data:
        vec_word = lang_data[lang]["vec_word"]
        vec_char = lang_data[lang]["vec_char"]
        mat_lang = lang_data[lang]["matrix"]
        ans_lang = lang_data[lang]["answers"]

        test_vec_word = vec_word.transform([test_text])
        test_vec_char = vec_char.transform([test_text])
        test_vec_lang = hstack([test_vec_word, test_vec_char])

        pred_lang, best_sim_lang = get_best_match(test_vec_lang, mat_lang, ans_lang)

    pred = pred_lang if best_sim_lang >= best_sim_global else pred_global
    predictions.append(str(pred) if pd.notna(pred) else "")


## === cell 3
submission_df = pd.DataFrame({"id": test_df["id"], "PredictionString": predictions})
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

submission_df.head()
