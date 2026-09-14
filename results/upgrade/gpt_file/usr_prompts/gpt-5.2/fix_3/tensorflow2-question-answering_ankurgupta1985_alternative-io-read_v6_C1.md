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

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

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

0.01628

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import json
import numpy as np
import pandas as pd

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:50]:
        pass
print("Example files under /kaggle/input (truncated):")
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        if filename.endswith(".jsonl") or filename.endswith(".csv"):
            print(os.path.join(dirname, filename))



## === cell 1
sample_path = "/kaggle/input/tensorflow2-question-answering/sample_submission.csv"
with open(sample_path, "r", encoding="utf-8") as f:
    n_lines = sum(1 for _ in f)
print(f"{n_lines} {sample_path}")




## === cell 2
def select_random_long_answer(long_answer_candidates, seed=None):
    assert isinstance(long_answer_candidates, list)
    for long_answer in long_answer_candidates:
        assert isinstance(long_answer, dict)
        assert "start_token" in long_answer
        assert "end_token" in long_answer
    if seed is not None:
        np.random.seed(seed)

    if len(long_answer_candidates) == 0:
        return {"start_token": 0, "end_token": 0}

    index = np.random.randint(len(long_answer_candidates))
    return long_answer_candidates[index]


def select_random_short_answer(long_answer, seed=None):
    assert isinstance(long_answer, dict)
    assert "start_token" in long_answer
    assert "end_token" in long_answer
    if seed is not None:
        np.random.seed(seed)

    if long_answer["end_token"] < long_answer["start_token"]:
        return {"start_token": 0, "end_token": 0}

    start_token = np.random.randint(
        low=long_answer["start_token"], high=long_answer["end_token"] + 1
    )
    end_token = np.random.randint(
        low=start_token, high=min(start_token + 10, long_answer["end_token"] + 1)
    )
    return {"start_token": start_token, "end_token": end_token}


def get_prediction_string(answer):
    assert isinstance(answer, dict)
    assert "start_token" in answer
    assert "end_token" in answer
    if answer["start_token"] == 0 and answer["end_token"] == 0:
        return ""
    return "{}:{}".format(answer["start_token"], answer["end_token"])


def get_answer_text(answer, document_text_tokens):
    assert isinstance(answer, dict)
    assert "start_token" in answer
    assert "end_token" in answer
    answer_tokens = document_text_tokens[
        answer["start_token"] : answer["end_token"] + 1
    ]
    answer_text = " ".join(answer_tokens)
    return answer_text


def _stable_int_seed_from_example_id(example_id, base_seed=42):
    s = str(example_id)
    h = 0
    for ch in s:
        h = (h * 131 + ord(ch)) % (2**32)
    return (h ^ np.uint32(base_seed)) % (2**32)


def predict_on_chunk_dataframe(df, seed=None):
    assert isinstance(df, pd.DataFrame)

    df = df.copy()
    df["document_text_tokens"] = df["document_text"].apply(lambda s: s.split())

    def _pick_long(row):
        row_seed = _stable_int_seed_from_example_id(
            row["example_id"], base_seed=seed if seed is not None else 42
        )
        return select_random_long_answer(
            row["long_answer_candidates"], seed=int(row_seed)
        )

    def _pick_short(row):
        row_seed = _stable_int_seed_from_example_id(
            row["example_id"], base_seed=(seed if seed is not None else 42) + 1
        )
        return select_random_short_answer(row["long_answer"], seed=int(row_seed))

    df["long_answer"] = df.apply(_pick_long, axis=1)
    df["short_answer"] = df.apply(_pick_short, axis=1)

    df["long_answer_text"] = df.apply(
        lambda row: get_answer_text(row["long_answer"], row["document_text_tokens"]),
        axis=1,
    )
    df["short_answer_text"] = df.apply(
        lambda row: get_answer_text(row["short_answer"], row["document_text_tokens"]),
        axis=1,
    )
    df["long_answer_prediction_string"] = df["long_answer"].apply(get_prediction_string)
    df["short_answer_prediction_string"] = df["short_answer"].apply(
        get_prediction_string
    )

    assert len(set(df.columns)) == len(df.columns)
    ordered_columns = [
        "question_text",
        "long_answer_text",
        "short_answer_text",
        "document_text",
    ]
    rest_columns = list(set(df.columns).difference(set(ordered_columns)))
    df = df[ordered_columns + rest_columns]
    return df


def generate_submission(df, seed=None):
    assert isinstance(df, pd.DataFrame)

    df = predict_on_chunk_dataframe(df, seed=seed)

    long_predictions = (
        df[["example_id", "long_answer_prediction_string"]]
        .copy()
        .rename({"long_answer_prediction_string": "PredictionString"}, axis=1)
    )
    long_predictions["example_id"] = (
        long_predictions["example_id"].astype(str) + "_long"
    )

    short_predictions = (
        df[["example_id", "short_answer_prediction_string"]]
        .copy()
        .rename({"short_answer_prediction_string": "PredictionString"}, axis=1)
    )
    short_predictions["example_id"] = (
        short_predictions["example_id"].astype(str) + "_short"
    )

    submission_df = pd.concat(
        [long_predictions, short_predictions], axis=0, ignore_index=True
    ).reset_index(drop=True)
    return submission_df




## === cell 3
test_path_candidates = [
    "/kaggle/input/tensorflow2-question-answering/simplified-nq-test.jsonl",
    "/kaggle/input/simplified-nq-test.jsonl",
]
test_path = None
for p in test_path_candidates:
    if os.path.exists(p):
        test_path = p
        break
if test_path is None:
    raise FileNotFoundError(
        f"Could not find simplified-nq-test.jsonl in expected locations: {test_path_candidates}"
    )

print("Using test file:", test_path)



## === cell 4
base_seed = 42
out_path = "submission.csv"

sample_sub = pd.read_csv(sample_path)
sample_ids = sample_sub["example_id"].astype(str)

with open(out_path, "w", encoding="utf-8") as out_f:
    out_f.write("example_id,PredictionString\n")

n_examples = 0
buffer = []
buffer_size = 256  # keep small for memory safety

with open(test_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        parsed_line = json.loads(line)
        buffer.append(parsed_line)
        if len(buffer) >= buffer_size:
            chunk_test_df = pd.DataFrame.from_records(buffer)
            sub_chunk = generate_submission(chunk_test_df, seed=base_seed)
            sub_chunk.to_csv(out_path, mode="a", header=False, index=False)
            n_examples += len(buffer)
            buffer = []

if buffer:
    chunk_test_df = pd.DataFrame.from_records(buffer)
    sub_chunk = generate_submission(chunk_test_df, seed=base_seed)
    sub_chunk.to_csv(out_path, mode="a", header=False, index=False)
    n_examples += len(buffer)

print("Number of test examples read =", n_examples)

sub = pd.read_csv(out_path)
sub["example_id"] = sub["example_id"].astype(str)
sub = sub.drop_duplicates(subset=["example_id"], keep="last")

sub = sample_sub[["example_id"]].merge(sub, on="example_id", how="left")
sub["PredictionString"] = sub["PredictionString"].fillna("")
sub.to_csv(out_path, index=False)

print("Final submission rows:", len(sub), "Expected:", len(sample_sub))



## === cell 5
submission_df_head = pd.read_csv(out_path).head()
print(submission_df_head)



## === cell 6
with open("n-examples.csv", "w", encoding="utf-8") as f:
    f.writelines(str(n_examples))

print("submission.csv size (bytes):", os.path.getsize(out_path))



## === cell 7
with open(out_path, "r", encoding="utf-8") as f:
    line_count = sum(1 for _ in f)
print("submission.csv line count:", line_count, "(includes header)")



## === cell 8
with open(out_path, "r", encoding="utf-8") as f:
    for _ in range(10):
        print(f.readline().rstrip("\n"))
