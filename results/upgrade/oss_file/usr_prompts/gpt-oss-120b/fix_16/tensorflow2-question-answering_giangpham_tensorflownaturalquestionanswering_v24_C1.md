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

3.10

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

0.4644976468180888

# 6. Current score

0.37371

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.03143) has done: 'Implemented a clean end‑to‑end pipeline that skips the problematic TensorFlow‑BERT imports, reads the test JSONL file directly, extracts the longest candidate span for each example, and builds a proper submission CSV with both “_long” and “_short” rows. All syntax errors and stray markdown have been removed, and the script now writes `submission.csv` in the required format without runtime failures. This minimal yet functional approach ensures a valid submission while providing a reasonable heuristic to move the score toward the target.'
- What this solution (achieved 0.06894) has done: 'I add a lightweight heuristic that selects, for each example, the candidate long answer whose text shares the most words with the question (fallback to the longest span). This keeps the overall pipeline unchanged while giving predictions that better match the ground‑truth spans, which should raise the micro‑F1 toward the target. The rest of the script (reading the test file, building the submission DataFrame, and saving it) remains the same.'
- What this solution (achieved 0.32601) has done: 'We improve the heuristic by selecting the long‑answer candidate with the highest overlap‑to‑length ratio (instead of raw overlap) and keep the fallback to the longest span when no overlap exists. For the short answer we output an empty string, avoiding false short predictions that hurt the micro‑F1. The rest of the pipeline stays unchanged, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.32527) has done: 'We keep the overall pipeline unchanged but add a tiny heuristic for short answers: when the question contains the word “yes” or “no” and the selected long‑answer candidate also contains that word, we output “YES” or “NO” respectively (otherwise we keep the short field blank). This small change is expected to raise the micro‑F1 toward the target without altering the core model logic.'
- What this solution (achieved 0.32408) has done: 'The fix replaces the tuple sorting that caused a TypeError by using a custom key that avoids comparing dictionaries, and improves the candidate scoring to use an overlap‑to‑length ratio (with tie‑breakers) which should raise the micro‑F1 toward the target while keeping the original pipeline unchanged. The rest of the code, including file handling and submission formatting, remains the same.'
- What this solution (achieved 0.37398) has done: 'I tighten the long‑answer scoring so that pure token overlap is prioritized (which better matches the ground‑truth spans) and keep the overlap‑to‑length ratio only as a secondary tie‑breaker. The short‑answer heuristic is refined to emit “YES” or “NO” only when the selected long answer actually contains that word, reducing false positives. These small tweaks stay within the original pipeline while expected to raise the micro‑F1 toward the target.'
- What this solution (achieved 0.05418) has done: 'I add a simple stop‑word filter to focus the overlap scoring on informative tokens and, when no YES/NO short answer is detected, copy the chosen long‑answer span as the short answer. These tiny tweaks keep the original pipeline unchanged while giving a modest boost in micro‑F1 toward the target.'
- What this solution (achieved 0.37398) has done: 'I restore the more effective overlap‑based heuristic, stop using the stop‑word filter (so the question tokens retain all words), remove the overly aggressive short‑answer copy‑long‑answer fallback, and drop the unnecessary tqdm import for robustness. These tweaks keep the original pipeline but are expected to raise the micro‑F1 substantially toward the target score.'
- What this solution (achieved 0.05409) has done: 'I refine the heuristics while keeping the overall pipeline unchanged:  
1. Filter stop‑words out of the question tokens (and later the candidate tokens) so overlap focuses on meaningful words.  
2. If the short‑answer heuristic does not produce YES/NO, reuse the selected long‑answer span as the short answer (when a span exists). This adds likely correct short answers without altering the core logic.  

These modest tweaks should raise the micro F1 toward the target while preserving the existing structure.'
- What this solution (achieved 0.37398) has done: 'I improve the heuristic by (1) removing the stop‑word filtering so the overlap is computed on the full token sets – this matches the earlier version that achieved a much higher micro‑F1, and (2) changing the short‑answer fallback to leave the field blank instead of copying the long‑answer span, which avoids many incorrect short predictions. These minimal adjustments keep the overall pipeline unchanged while moving the score much closer to the target.'
- What this solution (achieved 0.37397) has done: 'I keep the overall pipeline unchanged but modify the fallback when no candidate shares any token with the question: instead of picking the longest span (which often adds noise), the script now choose the shortest candidate. This reduces the chance of predicting overly large incorrect answers, which should improve the micro‑F1 and move the score closer to the target.'
- What this solution (achieved 0.37398) has done: 'I adjust the heuristic to favor longer candidate spans when overlaps tie (by using a positive length key) and, when no token overlap exists, fall back to the longest span instead of the shortest. These tweaks keep the overall pipeline untouched while giving the model a better chance of matching the true answer boundaries, which should raise the micro‑F1 toward the target score.'
- What this solution (achieved 0.37371) has done: 'I adjust the overlap scoring to ignore common stop‑words, making the token‑overlap metric more focused on informative words. This small change keeps the overall pipeline unchanged, only refines how candidates are ranked, and is expected to increase the micro‑F1 toward the target score.'

# 9. Code solution

## === cell 0
import os
import json
import pandas as pd

_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "in",
    "on",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "out",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "can",
    "will",
    "just",
    "don",
    "should",
    "now",
}




## === cell 1
def load_test_instances(test_path):
    """
    Reads the test JSONL and creates heuristic predictions:
    - tokenises document and question (removes stop‑words from overlap calculation)
    - scores each long_answer_candidate primarily by raw token overlap with the question,
      then by overlap‑to‑length ratio, and finally prefers longer candidates.
    - selects the best candidate; if no overlap, falls back to the longest span.
    - builds the long answer as start:end.
    - short answer heuristic:
        * output YES if both question and selected long answer contain "yes"
        * output NO  if both contain "no"
        * otherwise leave the short field blank
    Returns a list of dicts with example_id, long and short strings.
    """
    instances = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)

            eid = str(data.get("example_id"))
            doc_text = data.get("document_text", "")
            question = data.get("question_text", "")

            doc_tokens = doc_text.split()
            question_tokens = {
                w for w in question.lower().split() if w not in _STOPWORDS
            }

            candidates = data.get("long_answer_candidates", [])
            best_pred = ""
            short_pred = ""  # default: blank short answer

            if candidates:

                def candidate_text(c):
                    start = c.get("start_token")
                    end = c.get("end_token")
                    if start is None or end is None:
                        return ""
                    return " ".join(doc_tokens[start:end])

                scored = []
                for c in candidates:
                    txt = candidate_text(c).lower()
                    cand_tokens = [w for w in txt.split() if w not in _STOPWORDS]
                    overlap = len(set(cand_tokens) & question_tokens)
                    length = len(cand_tokens)
                    ratio = overlap / length if length > 0 else 0.0
                    scored.append((overlap, ratio, length, c, txt))

                best_overlap, best_ratio, _, best_candidate, best_text = max(
                    scored, key=lambda x: (x[0], x[1], x[2])
                )

                if best_overlap == 0:
                    best_candidate = max(
                        candidates,
                        key=lambda c: (
                            (c.get("end_token", 0) or 0)
                            - (c.get("start_token", 0) or 0)
                        ),
                    )
                    best_text = candidate_text(best_candidate).lower()
                    best_overlap = len(set(best_text.split()) & question_tokens)

                start = best_candidate.get("start_token")
                end = best_candidate.get("end_token")
                if start is not None and end is not None:
                    best_pred = f"{start}:{end}"

                if "yes" in question_tokens and "yes" in best_text.split():
                    short_pred = "YES"
                elif "no" in question_tokens and "no" in best_text.split():
                    short_pred = "NO"
                else:
                    short_pred = ""  # keep blank instead of copying long span

            instances.append(
                {"example_id": eid, "long": best_pred, "short": short_pred}
            )
    return instances




## === cell 2
possible_paths = [
    "../input/tensorflow2-question-answering/simplified-nq-test.jsonl",
    "../input/simplified-nq-test.jsonl",
    "./simplified-nq-test.jsonl",
]
test_path = next((p for p in possible_paths if os.path.exists(p)), None)
if test_path is None:
    raise FileNotFoundError(
        "Test file not found. Checked paths: " + ", ".join(p for p in possible_paths)
    )

instances = load_test_instances(test_path)

rows = []
for ins in instances:
    rows.append(
        {"example_id": f"{ins['example_id']}_long", "PredictionString": ins["long"]}
    )
    rows.append(
        {"example_id": f"{ins['example_id']}_short", "PredictionString": ins["short"]}
    )

submission = pd.DataFrame(rows, columns=["example_id", "PredictionString"])

output_path = "./submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path} with {len(submission)} rows.")
