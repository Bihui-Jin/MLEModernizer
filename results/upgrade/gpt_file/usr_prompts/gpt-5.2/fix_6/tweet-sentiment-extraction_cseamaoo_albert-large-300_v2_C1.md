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
Predict the word or phrase from tweets that exemplifies the labelled sentiment.

## Metric
Word-level Jaccard score.

## Submission Format
For each ID in the test set, you must predict the string that best supports the sentiment for the tweet in question. Note that the selected text _needs_ to be **quoted** and **complete** (include punctuation, etc. - the above code splits ONLY on whitespace) to work correctly. The file should contain a header and have the following format:
```
textID,selected_text
2,"very good"
5,"I don't care"
6,"bad"
8,"it was, yes"
etc.
```

## Dataset
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

- `textID` - unique ID for each piece of text
- `text` - the text of the tweet
- `sentiment` - the general sentiment of the tweet
- `selected_text` - [train only] the text that supports the tweet's sentiment

# 2. Python version

3.8

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (130 lines)
            sample_submission.csv (2750 lines)
            sample_submission.csv.zip (18.6 kB)
            test.csv (2750 lines)
            test.csv.zip (114.6 kB)
            train.csv (24733 lines)
            train.csv.zip (1.2 MB)
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
        input/
            description.md (130 lines)
            sample_submission.csv (2750 lines)
            sample_submission.csv.zip (18.6 kB)
            test.csv (2750 lines)
            test.csv.zip (114.6 kB)
            train.csv (24733 lines)
            train.csv.zip (1.2 MB)
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
        working/
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
```

-> data/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> data/test.csv has 2749 rows and 3 columns.
The columns are: textID, text, sentiment

-> data/train.csv has 24732 rows and 4 columns.
The columns are: textID, text, selected_text, sentiment

-> data/tweet-sentiment-extraction/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> data/tweet-sentiment-extraction/test.csv has 2749 rows and 3 columns.
The columns are: textID, text, sentiment

-> data/tweet-sentiment-extraction/train.csv has 24732 rows and 4 columns.
The columns are: textID, text, selected_text, sentiment

-> input/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> (stopped after 10 files for performance)

# 5. Target score

0.7034798860549927

# 6. Current score

0.62933

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.55421) has done: 'I remove the hard failure on the missing `/kaggle/input/albert-xl-300/` directory and instead run a lightweight, deterministic fallback that still produces a valid `submission.csv`. I also fix the protobuf/transformers import/runtime issue that prevents `run_prediction` from being defined by guarding the transformer QA path and only using it when a local model is actually available. Finally, I ensure the submission is correctly aligned to `sample_submission.csv` and that `selected_text` is always a string (quoted properly by CSV), so Kaggle accepts the file. This should run end-to-end in the provided environment and yield a non-trivial baseline score (typically around the target band for this competition) without changing the overall intent of extracting a sentiment-supporting span.'
- What this solution (achieved 0.59096) has done: 'Your current score (0.55421) is well below the target (0.70348), so we should improve extraction quality without changing the overall “span from tweet based on sentiment” core approach. The biggest low-risk gain is to replace the single-token heuristic with a deterministic phrase-span heuristic that expands around sentiment cue words, handles common negations, and (when no cue exists) selects a high-sentiment window rather than one word. Neutral handling stays the same (return full tweet), preserving evaluation semantics and keeping the pipeline lightweight and fully offline. The submission writing/alignment logic is kept intact, so it still produces a valid `submission.csv`.'
- What this solution (achieved 0.61332) has done: 'We’re still well below the target (0.59096 vs 0.70348), so the smallest safe way to move upward is to improve the heuristic span extraction without changing the overall “extract a substring span based on sentiment” approach. I keep your deterministic cue-based windowing, but add two low-risk fixes that typically boost Jaccard: (1) return the exact substring from the original tweet using character offsets (prevents punctuation/quote mismatches), and (2) expand/contract around the best window using sentiment-consistent token scoring rather than a fixed window size. Neutral remains “return full tweet”, and submission alignment/writing stays the same so it still produces a valid `submission.csv`.'
- What this solution (achieved 0.62933) has done: 'Your current score (0.61332) is still far below the target (0.70348), so we should improve the heuristic span extractor while keeping the same overall “extract a substring span from the tweet conditioned on sentiment” approach. The biggest low-risk gain is to better handle common Twitter patterns that the Jaccard metric is sensitive to: (1) make the tokenizer/normalizer recognize more sentiment cues (including slang/emoji and elongated words), and (2) expand spans across conjunctions/particles that typically belong to the opinion phrase (“so”, “very”, “not”, “but”, etc.) while still returning an exact substring via offsets. I also add a tiny rule for very short tweets (1–2 tokens) and for cases where the gold answer is usually the whole tweet for positive/negative (common in this dataset when the whole tweet is just an opinion). Submission writing stays identical and aligned to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os

is_inference_flag = False
model_dir = "/kaggle/input/albert-xl-300/"
try:
    tweet_models_dir = os.listdir(model_dir)
    if len(tweet_models_dir) > 0:
        is_inference_flag = True
except Exception:
    is_inference_flag = False



## === cell 1
print("Inference flag status :", is_inference_flag)
if not is_inference_flag:
    print(
        "WARNING: Inference model directory /kaggle/input/albert-xl-300/ not found. "
        "Falling back to a deterministic span-extraction heuristic."
    )



## === cell 2
import pandas as pd
import numpy as np
import json
import re
from sklearn.model_selection import train_test_split



## === cell 3
train_df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
test_df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
sub_df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")




## === cell 4
def jaccard(str1, str2):
    a = str(str1).lower().split()
    b = str(str2).lower().split()
    c = set(a).intersection(set(b))
    denom = len(a) + len(b) - len(c)
    if denom == 0:
        return 0.0
    return float(len(c)) / denom




## === cell 5
train_df.dropna(inplace=True)



## === cell 6
X_train, X_test = train_test_split(train_df, test_size=0.10, random_state=42)



## === cell 7
train = np.array(X_train)
val = np.array(X_test)
test = np.array(test_df)
use_cuda = True



## === cell 8
os.makedirs("data/models/albert", exist_ok=True)
os.makedirs("data/models/roberta", exist_ok=True)



## === cell 9
"""
Prepare training data in QA-compatible format (kept for completeness).
In this environment we may run inference-only, so we do not write train/val json.
"""


def find_all(input_str, search_str):
    l1 = []
    length = len(input_str)
    index = 0
    while index < length:
        i = input_str.find(search_str, index)
        if i == -1:
            return l1
        l1.append(i)
        index = i + 1
    return l1


def do_qa_train(train_arr):
    output = {}
    output["version"] = "v1.0"
    output["data"] = []
    paragraphs = []
    for line in train_arr:
        context = line[1]
        qas = []
        question = line[-1]
        qid = line[0]
        answers = []
        answer = line[2]
        if type(answer) != str or type(context) != str or type(question) != str:
            continue
        answer_starts = find_all(context, answer)
        for answer_start in answer_starts:
            answers.append({"answer_start": answer_start, "text": answer.lower()})
            break
        qas.append(
            {
                "question": question,
                "id": qid,
                "is_impossible": False,
                "answers": answers,
            }
        )
        paragraphs.append({"context": context.lower(), "qas": qas})

    output["data"].append({"title": "None", "paragraphs": paragraphs})
    return output




## === cell 10
"""
Prepare testing data in QA-compatible format
"""


def convert_test_qa_json(test_arr):
    output = {}
    output["version"] = "v1.0"
    output["data"] = []
    paragraphs = []
    for line in test_arr:
        context = line[1]
        qas = []
        question = line[-1]
        qid = line[0]
        if type(context) != str or type(question) != str:
            continue
        answers = []
        answers.append({"answer_start": 1000000, "text": "__None__"})
        qas.append(
            {
                "question": question,
                "id": qid,
                "is_impossible": False,
                "answers": answers,
            }
        )
        paragraphs.append({"context": context.lower(), "qas": qas})

    output["data"].append({"title": "None", "paragraphs": paragraphs})
    return output


qa_test = convert_test_qa_json(test)
with open("data/test.json", "w") as outfile:
    json.dump(qa_test, outfile)



## === cell 11
pass



## === cell 12
pass



## === cell 13
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")



## === cell 14
USE_QA_MODEL = False
run_prediction = None


def _clean_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _normalize_token_for_match(tok: str) -> str:
    t = tok.lower()
    t = re.sub(r"^[^\w']+|[^\w']+$", "", t)  # strip edge punct but keep apostrophes
    if not t:
        return ""
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    t = t.replace("dont", "don't").replace("cant", "can't").replace("wont", "won't")
    return t


def _tokenize_with_offsets(text: str):
    return [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def _heuristic_selected_text(sentiment: str, text: str) -> str:
    if not isinstance(text, str):
        text = ""
    if not isinstance(sentiment, str):
        sentiment = "neutral"

    text_stripped = _clean_whitespace(text)
    if text_stripped == "":
        return ""

    s = sentiment.lower().strip()

    if s == "neutral":
        return text_stripped

    toks = _tokenize_with_offsets(text_stripped)
    orig_tokens = [t[0] for t in toks]
    offsets = [(t[1], t[2]) for t in toks]
    low_tokens = [_normalize_token_for_match(t) for t in orig_tokens]
    n = len(orig_tokens)
    if n == 0:
        return ""

    if n <= 2:
        return text_stripped

    pos_words = {
        "love",
        "loved",
        "lovely",
        "like",
        "liked",
        "good",
        "great",
        "best",
        "amazing",
        "awesome",
        "wonderful",
        "nice",
        "happy",
        "glad",
        "fantastic",
        "excellent",
        "perfect",
        "enjoy",
        "enjoyed",
        "yay",
        "beautiful",
        "excited",
        "fun",
        "cool",
        "sweet",
        "thank",
        "thanks",
        "thx",
        "congrats",
        "congratulations",
        "brilliant",
        "fabulous",
        "goodnight",
        "goodmorning",
    }
    neg_words = {
        "hate",
        "hated",
        "awful",
        "bad",
        "worst",
        "terrible",
        "sad",
        "angry",
        "mad",
        "annoying",
        "disappointed",
        "disappointing",
        "sucks",
        "suck",
        "ugh",
        "horrible",
        "poor",
        "boring",
        "gross",
        "tired",
        "sick",
        "pain",
        "cry",
        "sorry",
        "miss",
        "missing",
        "wtf",
        "damn",
    }
    negators = {
        "not",
        "no",
        "never",
        "n't",
        "don't",
        "didn't",
        "can't",
        "won't",
        "cannot",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "haven't",
        "hasn't",
        "hadn't",
    }
    intensifiers = {
        "very",
        "so",
        "too",
        "really",
        "super",
        "quite",
        "extremely",
        "absolutely",
        "totally",
        "pretty",
        "highly",
        "much",
        "soo",  # from normalization
    }
    joiners = {
        "and",
        "or",
        "but",
        "because",
        "cuz",
        "bc",
        "though",
        "tho",
        "!",
        "!!",
        "...",
    }

    target = pos_words if s == "positive" else neg_words
    opposite = neg_words if s == "positive" else pos_words

    scores = np.zeros(n, dtype=np.float32)
    for i, w in enumerate(low_tokens):
        if not w:
            continue

        if w in target:
            scores[i] += 2.2
        if w in opposite:
            scores[i] -= 1.2

        if i > 0 and low_tokens[i - 1] in negators and w in opposite:
            scores[i] += 2.2
            scores[i - 1] += 0.8
        if i > 0 and low_tokens[i - 1] in negators and w in target:
            scores[i] -= 2.2
            scores[i - 1] += 0.6

        if (
            i > 0
            and low_tokens[i - 1] in intensifiers
            and (w in target or w in opposite)
        ):
            scores[i] += 0.8
            scores[i - 1] += 0.3

        ot = orig_tokens[i].lower()
        if ot in {":)", ":-)", ":d", ":-d", "<3", "❤", "♥"} and s == "positive":
            scores[i] += 1.8
        if ot in {":(", ":-(", ":'(", ":'-(", "d:", "☹"} and s == "negative":
            scores[i] += 1.8

    if float(scores.max()) > 0.0:
        topk = min(6, n)
        centers = list(np.argsort(-scores)[:topk])

        best_l, best_r = 0, 1
        best_obj = -1e18

        for center in centers:
            for l in range(max(0, center - 12), center + 1):
                for r in range(center + 1, min(n, center + 13) + 1):
                    span_scores = scores[l:r]
                    span_sum = float(span_scores.sum())
                    length = r - l
                    obj = span_sum - 0.10 * length
                    if (
                        l > 0
                        and low_tokens[l] in target
                        and low_tokens[l - 1] in negators
                    ):
                        obj += 0.15
                    if obj > best_obj:
                        best_obj = obj
                        best_l, best_r = l, r

        l, r = best_l, best_r

        while l > 0 and low_tokens[l - 1] in (negators | intensifiers):
            l -= 1

        while r < n and (
            low_tokens[r] in joiners or orig_tokens[r] in {"!", "!!", "..."}
        ):
            r += 1

        start = offsets[l][0]
        end = offsets[r - 1][1]
        pred = text_stripped[start:end].strip()

        if len(pred) >= 0.85 * len(text_stripped):
            return text_stripped
        return pred

    stop = {
        "i",
        "a",
        "an",
        "the",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "for",
        "is",
        "it",
        "this",
        "that",
        "was",
        "are",
        "am",
        "im",
        "i'm",
        "we",
        "you",
        "u",
        "me",
        "my",
        "your",
        "with",
        "at",
        "be",
        "been",
        "have",
        "has",
        "had",
        "as",
        "they",
        "them",
        "he",
        "she",
        "his",
        "her",
    }

    def content_score(tok_norm: str) -> float:
        if not tok_norm or tok_norm in stop:
            return 0.0
        if any(ch.isdigit() for ch in tok_norm):
            return 0.7
        if len(tok_norm) >= 8:
            return 1.2
        return 0.8

    content = np.array([content_score(w) for w in low_tokens], dtype=np.float32)
    if n == 1:
        return orig_tokens[0]

    best_sum = -1e9
    best_l, best_r = 0, min(n, 4)
    for w in (3, 4, 5, 6):
        if n < w:
            continue
        for l in range(0, n - w + 1):
            r = l + w
            ssum = float(content[l:r].sum())
            if ssum > best_sum:
                best_sum = ssum
                best_l, best_r = l, r

    start = offsets[best_l][0]
    end = offsets[best_r - 1][1]
    return text_stripped[start:end].strip()


if is_inference_flag and os.path.isdir(model_dir):
    try:
        import torch
        from torch.utils.data import DataLoader, SequentialSampler
        from collections import OrderedDict

        from transformers import (
            AutoConfig,
            AutoModelForQuestionAnswering,
            AutoTokenizer,
            squad_convert_examples_to_features,
        )
        from transformers.data.processors.squad import SquadResult, SquadExample
        from transformers.data.metrics.squad_metrics import compute_predictions_logits

        model_name_or_path = model_dir
        output_dir = ""

        n_best_size = 1
        max_answer_length = 254
        do_lower_case = True
        null_score_diff_threshold = 0.0

        def to_list(tensor):
            return tensor.detach().cpu().tolist()

        config_class, model_class, tokenizer_class = (
            AutoConfig,
            AutoModelForQuestionAnswering,
            AutoTokenizer,
        )

        config = config_class.from_pretrained(model_name_or_path)
        tokenizer = tokenizer_class.from_pretrained(
            model_name_or_path, do_lower_case=True
        )
        model = model_class.from_pretrained(model_name_or_path, config=config)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        def run_prediction(question_texts, context_text):
            """Setup function to compute predictions (QA model path)."""
            if question_texts[0] != "neutral":
                examples = []
                for i, question_text in enumerate(question_texts):
                    example = SquadExample(
                        qas_id=str(i),
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=None,
                        start_position_character=None,
                        title="Predict",
                        is_impossible=False,
                        answers=None,
                    )
                    examples.append(example)

                features, dataset = squad_convert_examples_to_features(
                    examples=examples,
                    tokenizer=tokenizer,
                    max_seq_length=300,
                    doc_stride=128,
                    max_query_length=64,
                    is_training=False,
                    return_dataset="pt",
                    threads=1,
                )

                eval_sampler = SequentialSampler(dataset)
                eval_dataloader = DataLoader(
                    dataset, sampler=eval_sampler, batch_size=10
                )

                all_results = []
                for batch in eval_dataloader:
                    model.eval()
                    batch = tuple(t.to(device) for t in batch)
                    with torch.no_grad():
                        inputs = {
                            "input_ids": batch[0],
                            "attention_mask": batch[1],
                            "token_type_ids": batch[2],
                        }
                        example_indices = batch[3]
                        outputs = model(**inputs)

                        for j, example_index in enumerate(example_indices):
                            eval_feature = features[example_index.item()]
                            unique_id = int(eval_feature.unique_id)
                            output = [to_list(o[j]) for o in outputs]
                            start_logits, end_logits = output
                            result = SquadResult(unique_id, start_logits, end_logits)
                            all_results.append(result)

                output_prediction_file = "predictions.json"
                output_nbest_file = "nbest_predictions.json"
                output_null_log_odds_file = "null_predictions.json"

                predictions = compute_predictions_logits(
                    examples,
                    features,
                    all_results,
                    n_best_size,
                    max_answer_length,
                    do_lower_case,
                    output_prediction_file,
                    output_nbest_file,
                    output_null_log_odds_file,
                    False,  # verbose_logging
                    True,  # version_2_with_negative
                    null_score_diff_threshold,
                    tokenizer,
                )
            else:
                predictions = OrderedDict([(0, context_text)])

            return predictions

        USE_QA_MODEL = True
        print("QA model inference enabled from:", model_dir)

    except Exception as e:
        USE_QA_MODEL = False
        run_prediction = None
        print(
            "WARNING: QA model path detected but could not be initialized. Falling back to heuristic. Error:",
            repr(e),
        )

if not USE_QA_MODEL:
    from collections import OrderedDict

    def run_prediction(question_texts, context_text):
        sentiment = question_texts[0] if question_texts else "neutral"
        pred = _heuristic_selected_text(sentiment, context_text)
        return OrderedDict([(0, pred)])




## === cell 15
jaccard_scores = []
predictions_x_test = []

for _, row in X_test.head(10).iterrows():
    context = row["text"]
    selected_text = row["selected_text"]
    questions = [row["sentiment"]]
    preds_dict = run_prediction(questions, context)
    for key in preds_dict.keys():
        predicted_text = preds_dict[key]
        predictions_x_test.append(
            {
                "selected_text": selected_text,
                "predicted_text": predicted_text,
                "sentiment": row["sentiment"],
                "textID": row["textID"],
            }
        )
        jaccard_scores.append(jaccard(selected_text, predicted_text))

print(
    "Jaccard Score (sample of 10):",
    float(np.mean(jaccard_scores)) if len(jaccard_scores) else 0.0,
)
predictions_x_test_df = pd.DataFrame.from_dict(predictions_x_test)
print(predictions_x_test_df.head())



## === cell 16
print("test_df shape:", test_df.shape)



## === cell 17
predictions = []

for _, row in test_df.iterrows():
    context = row["text"]
    questions = [row["sentiment"]]
    preds_dict = run_prediction(questions, context)
    for key in preds_dict.keys():
        predicted_text = preds_dict[key]
        predictions.append({"textID": row["textID"], "selected_text": predicted_text})

predictions_df = pd.DataFrame.from_dict(predictions)

predictions_df = predictions_df.drop_duplicates(subset=["textID"], keep="first")
output_df = sub_df[["textID"]].merge(predictions_df, on="textID", how="left")

output_df["selected_text"] = output_df["selected_text"].fillna("").astype(str)

output_path = "submission.csv"
output_df.to_csv(output_path, index=False)

print(output_df.head())
print("Wrote:", output_path, "rows:", len(output_df), "cols:", list(output_df.columns))
print("QA model used:", USE_QA_MODEL)
