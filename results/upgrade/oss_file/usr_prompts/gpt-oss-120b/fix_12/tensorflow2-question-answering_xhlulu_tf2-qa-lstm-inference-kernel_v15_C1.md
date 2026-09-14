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
For each article + question pair, you must predict / select long and short form answers to the question drawn *directly from the article*.

- A long answer would be a longer section of text that answers the question - several sentences or a paragraph.
- A short answer might be a sentence or phrase, or even in some cases a YES/NO. The short answers are always contained within / a subset of one of the plausible long answers.
- A given article can (and very often will) allow for both long *and* short answers, depending on the question.

There is more detail about the data and what you're predicting [on the Github page for the Natural Questions dataset](https://github.com/google-research-datasets/natural-questions/blob/master/README.md). This page also contains helpful utilities and scripts. Note that we are using the simplified text version of the data - most of the HTML tags have been removed, and only those necessary to break up paragraphs / sections are included.

## Metric
Micro F1. Predicted long and short answers must match exactly the token indices of one of the ground truth labels ((or match YES/NO if the question has a yes/no short answer). There may be up to five labels for long answers, and more for short. If no answer applies, leave the prediction blank/null.

## Submission Format
For each ID in the test set, you must predict a) a set of start:end token indices, b) a YES/NO answer if applicable (short answers ONLY), or c) a BLANK answer if no prediction can be made. The file should contain a header and have the following format:

```
-7853356005143141653_long,6:18
-7853356005143141653_short,YES
-545833482873225036_long,105:200
-545833482873225036_short,
-6998273848279890840_long,
-6998273848279890840_short,NO
```
`
## Data
Each sample contains a Wikipedia article, a related question, and the candidate long form answers. The training examples also provide the correct long and short form answer or answers for the sample, if any exist.

- **simplified-nq-train.jsonl** - the training data, in newline-delimited JSON format.
- **simplified-nq-kaggle-test.jsonl** - the test data, in newline-delimited JSON format.
- **sample_submission.csv** - a sample submission file in the correct format

### Data fields
- **document_text** - the text of the article in question (with some HTML tags to provide document structure). The text can be tokenized by splitting on whitespace.
- **question_text** - the question to be answered
- **long_answer_candidates** - a JSON array containing all of the plausible long answers.
- **annotations** - a JSON array containing all of the correct long + short answers. Only provided for train.
- **document_url** - the URL for the full article. Provided for informational purposes only. This is NOT the simplified version of the article so indices from this cannot be used directly. The content may also no longer match the html used to generate document_text. Only provided for train.
- **example_id** - unique ID for the sample.

# 2. Python version

3.8

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

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

# 5. Target score

0.1718816973853407

# 6. Current score

0.06904

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.03227) has done: 'I fix the import error, remove the failing model/tokenizer loads, and replace the prediction step with a simple heuristic that selects the first candidate span for each example. This allows the script to run end‑to‑end, produce a correctly formatted `submission.csv`, and gives a baseline score that moves toward the target without altering the core architecture.'
- What this solution (achieved 0.03143) has done: 'The changes wrap TensorFlow imports in a safe try/except to avoid the protobuf error, load the sample submission using the correct absolute input path, and replace the “first‑candidate” baseline with a simple but better heuristic that picks the longest candidate span for each example (more likely to match the true answer). The rest of the pipeline remains unchanged, and the script now writes a properly formatted `submission.csv` ready for Kaggle.'
- What this solution (achieved 0.03143) has done: 'I add a small heuristic improvement: pick the longest candidate span for the long prediction (as before) but pick the *shortest* candidate span for the short prediction, which is more likely to match the true short answer. This changes only the post‑processing cells and keeps all core logic untouched, while fixing the earlier import issue already handled by the safe TensorFlow import wrapper. The script now produce a valid `submission.csv` and should achieve a higher micro‑F1 score, moving it closer to the target.'
- What this solution (achieved 0.06894) has done: 'I add a simple token‑overlap heuristic to choose candidate spans: for each example the long answer be the candidate with the highest overlap with the question, and the short answer be the candidate with the highest overlap preferring a shorter span. This replaces the previous longest/shortest heuristics while keeping the overall pipeline unchanged, fixing the low score and still producing a valid submission.csv.'
- What this solution (achieved 0.03896) has done: 'I adjust the scoring heuristic to better differentiate long and short answers: the long answer now prefers candidates with high overlap *and* longer spans, while the short answer prefers high overlap but penalizes longer spans by dividing by the span length. This small change keeps the overall pipeline intact, fixes no‑op issues, and is expected to raise the micro‑F1 toward the target. The rest of the script is left unchanged, and the final CSV is written correctly.'
- What this solution (achieved 0.03896) has done: 'I add the missing imports, replace the undefined `tqdm` with a simple loop, define a dummy `load_model` variable, and restructure the cells so that each step runs in order. The heuristics are slightly improved: the long answer now prefers candidates with high token overlap **and** longer span length, while the short answer keeps the overlap‑over‑length score and still handles YES/NO candidates. Finally, the script merges the predictions with the sample submission template (if it exists) and writes a correctly formatted `submission.csv`.'
- What this solution (achieved 0.03891) has done: 'I improve the token overlap calculation by stripping punctuation from tokens, which gives a more accurate similarity count, and adjust the long‑answer scoring to weight the span length directly (using `span_len` instead of `span_len+1`). These small changes keep the overall pipeline unchanged while expectedly increasing the micro‑F1 score toward the target.'
- What this solution (achieved 0.06904) has done: 'I adjust the heuristic scoring so that both long and short answer selections rely primarily on token overlap rather than being heavily weighted by span length. This change should increase the chance of picking the correct answer spans, moving the micro‑F1 closer to the target while keeping the overall pipeline unchanged. The modification is limited to the scoring lines in cell 4.'
- What this solution (achieved 0.0421) has done: 'I keep the overall pipeline unchanged but improve the heuristic scoring: for long answers I boost candidates that overlap more with the question **and** are slightly longer, while for short answers I favor higher overlap per token (overlap divided by span length). This simple change should raise the micro‑F1 toward the target without altering any core logic or I/O.'
- What this solution (achieved 0.06904) has done: 'I adjust the heuristic scoring so that the long‑answer selection no longer favors overly long spans (by removing the length boost) while keeping the short‑answer scoring unchanged. This small change is expected to pick more accurate long answers, thereby raising the micro‑F1 score toward the target without altering the core pipeline.'

# 9. Code solution

## === cell 0
import json
import pandas as pd
import pickle
import string

load_model = None




## === cell 1
def build_test(test_path):
    """Parse the test jsonl and create a DataFrame with one row per candidate."""
    processed_rows = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            text_tokens = line["document_text"].split()
            question = line["question_text"]
            example_id = line["example_id"]
            for candidate in line["long_answer_candidates"]:
                start = candidate["start_token"]
                end = candidate["end_token"]
                span_str = f"{start}:{end}"
                span_len = end - start
                processed_rows.append(
                    {
                        "example_id": example_id,
                        "PredictionString": span_str,
                        "span_len": span_len,
                        "text": " ".join(text_tokens[start:end]),
                        "question": question,
                    }
                )
    return pd.DataFrame(processed_rows)


directory = "/kaggle/input/tensorflow2-question-answering/"
test_path = directory + "simplified-nq-test.jsonl"
sample_submission_path = directory + "sample_submission.csv"

test = build_test(test_path)

try:
    submission = pd.read_csv(
        sample_submission_path, dtype={"example_id": str, "PredictionString": str}
    )
except Exception:
    submission = pd.DataFrame(columns=["example_id", "PredictionString"])



## === cell 2
if load_model is not None:
    try:
        model = load_model("/kaggle/input/tf-qa-new-start/model.h5")
    except Exception:
        model = None
else:
    model = None  # TensorFlow not available; we rely on the heuristic baseline



## === cell 3
try:
    with open("/kaggle/input/tf-qa-new-start/tokenizer.pickle", "rb") as f:
        tokenizer = pickle.load(f)
except Exception:
    tokenizer = None  # Not required for the heuristic baseline




## === cell 4
def _clean_tokens(text):
    return set(
        tok.strip(string.punctuation).lower()
        for tok in text.split()
        if tok.strip(string.punctuation)
    )


def _overlap(row):
    q_tokens = _clean_tokens(row["question"])
    t_tokens = _clean_tokens(row["text"])
    return len(q_tokens.intersection(t_tokens))


test["overlap"] = test.apply(_overlap, axis=1)

test["score_long"] = test["overlap"]

test["score_short"] = test["overlap"] / (test["span_len"] + 1)

idx_long = test.groupby("example_id")["score_long"].idxmax()
result_long = test.loc[idx_long, ["example_id", "PredictionString"]].reset_index(
    drop=True
)

yesno_mask = test["text"].str.strip().str.lower().isin(["yes", "no"])
yesno_candidates = test[yesno_mask].copy()

idx_short = test.groupby("example_id")["score_short"].idxmax()
result_short = test.loc[idx_short, ["example_id", "PredictionString"]].copy()

if not yesno_candidates.empty:
    yesno_best = yesno_candidates.groupby("example_id").first().reset_index()
    yesno_best["PredictionString"] = yesno_best["text"].str.strip().str.upper()
    result_short = result_short.merge(
        yesno_best[["example_id", "PredictionString"]],
        on="example_id",
        how="left",
        suffixes=("", "_yesno"),
    )
    result_short["PredictionString"] = result_short[
        "PredictionString_yesno"
    ].combine_first(result_short["PredictionString"])
    result_short = result_short[["example_id", "PredictionString"]]



## === cell 5
result_long = result_long.assign(example_id=lambda df: df["example_id"] + "_long")
result_short = result_short.assign(example_id=lambda df: df["example_id"] + "_short")
result_combined = pd.concat([result_long, result_short], ignore_index=True)



## === cell 6
if not submission.empty:
    final_submission = submission.drop(columns=["PredictionString"]).merge(
        result_combined, on="example_id", how="left"
    )
else:
    final_submission = result_combined.copy()



## === cell 7
final_submission.to_csv("submission.csv", index=False)
