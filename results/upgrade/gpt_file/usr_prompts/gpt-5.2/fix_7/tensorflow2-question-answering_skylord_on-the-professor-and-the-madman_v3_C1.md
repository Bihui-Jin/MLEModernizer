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

beautifulsoup4==4.13.4
geopandas==0.14.4
nltk==3.9.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

0.10091

# 6. Current score

0.06461

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05706) has done: 'The timeout is dominated by loading the entire 1.7GB test JSONL into a DataFrame and then repeatedly doing slow per-row `.loc` access, repeatedly rebuilding stopwords, and parsing HTML with BeautifulSoup for every example. To keep core logic identical, the optimized script streams the test JSONL line-by-line (no full DataFrame), uses `html.parser` directly for just `<p>` extraction (equivalent to the BeautifulSoup `find_all('p')` usage here), precomputes the stopword set once, and replaces DataFrame row access with local variables and list appends. This preserves the same prediction semantics (same random/trigger behavior and same span selection logic), while removing the major memory/time overhead. It also speeds up submission creation by mapping base example_id directly rather than per-row `.loc` access.'
- What this solution (achieved 0.07647) has done: 'To move your score up toward 0.10091 with minimal disruption, I keep your exact heuristic pipeline but add one conservative “no-answer gating” step learned from the 2000-row train sample you already load: only emit a long/short span when the best paragraph-vs-question word overlap is above a small threshold; otherwise output blanks (which reduces false positives and typically improves micro-F1 on NQ). This keeps the same paragraph selection logic (`qa_word_match`) and token-span construction; it only decides whether to abstain. I also make the long span come from the selected long answer candidate boundaries when possible (still driven by the same chosen paragraph), which increases exact-match probability without changing the overall approach. The rest (streaming test JSONL, stopwords cache, yes/no trigger behavior) remains unchanged.'
- What this solution (achieved 0.05781) has done: 'To move your score up toward 0.10091 (from 0.07647) while keeping the same heuristic core, I make two minimal, metric-aligned changes: (1) keep the existing overlap-based abstention, but tune the overlap threshold more conservatively by learning it from train via a tiny grid search that directly maximizes micro-F1 on the 2000-row train subset; (2) when we do predict, snap the long span to the best-matching *top-level* long answer candidate using token-overlap with the question (still the same word-overlap idea), which improves exact token-index match probability. Everything else (paragraph extraction, yes/no behavior, random short spans, streaming test JSONL, submission formatting) remains the same to preserve semantics and runtime safety. This should increase precision/recall balance and typically lift micro-F1 toward the target without introducing new modeling logic. The submission file remains `submission.csv` with the required columns.'
- What this solution (achieved 0.05999) has done: 'I make two minimal, metric-aligned tweaks that should lift micro-F1 toward your target without changing the overall heuristic approach: first, tune the abstention threshold on train using a micro-F1 proxy that matches the competition (exact token-span match against one ground-truth long answer), rather than the current “has any long label” proxy. Second, when we do predict, I snap the long span to the best candidate using paragraph-token overlap (still the same overlap idea), with a safe fallback to your current question-overlap candidate snap; this increases the chance of exact candidate boundary matches. Everything else (paragraph extraction, random YES/NO and short-span behavior, streaming test, output format) is preserved.'
- What this solution (achieved 0.06461) has done: 'I keep your heuristic pipeline unchanged but make two metric-aligned tweaks to move micro-F1 up toward the 0.10091 target: (1) tune the overlap abstention threshold on train using a stricter proxy that requires *any* gold long span match among up to 5 annotations (not just the first), which reduces false “FP” during tuning and yields a more reliable threshold; (2) when snapping to a long-answer candidate, score candidates by question-overlap with light token normalization (strip punctuation + lowercase) to better match token boundaries and improve exact candidate selection without changing the overall approach. These are small, localized changes that preserve your paragraph selection, snapping strategy, yes/no behavior, and submission formatting. The script still runs end-to-end and writes `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os
import json
import random
import numpy as np
import pandas as pd

random.seed(123)
np.random.seed(123)




## === cell 1
def read_lines_m(path, max_limit=None):
    """
    Read jsonl into DataFrame.
    """
    rows = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if max_limit is not None and i >= max_limit:
                break
            rows.append(json.loads(line))
    return pd.DataFrame(rows)




## === cell 2
p = "/kaggle/input/tensorflow2-question-answering/"



## === cell 3
train = read_lines_m(os.path.join(p, "simplified-nq-train.jsonl"), max_limit=2000)

print(train.shape)
print(train.columns)



## === cell 4
if len(train) > 0:
    print(train.question_text.iloc[0])



## === cell 5
if len(train) > 0:
    print(train.annotations.iloc[0])



## === cell 6
if len(train) > 0:
    ann0 = train.annotations.iloc[0][0]
    la = ann0["long_answer"]
    if la["start_token"] >= 0 and la["end_token"] > la["start_token"]:
        toks = train.document_text.iloc[0].split()
        print(" ".join(toks[la["start_token"] : la["end_token"]])[:500])



## === cell 7
if len(train) > 0:
    ann0 = train.annotations.iloc[0][0]
    if len(ann0.get("short_answers", [])) > 0:
        sa0 = ann0["short_answers"][0]
        toks = train.document_text.iloc[0].split()
        print(" ".join(toks[sa0["start_token"] : sa0["end_token"]])[:500])
    else:
        print("No short answer in this example.")



## === cell 8
if len(train) > 105:
    print(train.annotations.iloc[105])



## === cell 9
if len(train) > 105:
    i = 105
    ann = train.annotations.iloc[i][0]
    toks = train.document_text.iloc[i].split()

    print(train.question_text.iloc[i])
    print("Long Answer:")
    la = ann["long_answer"]
    if la["start_token"] >= 0 and la["end_token"] > la["start_token"]:
        print(" ".join(toks[la["start_token"] : la["end_token"]])[:500])
    else:
        print("No long answer.")

    print("Short Answer:")
    if len(ann.get("short_answers", [])) > 0:
        sa = ann["short_answers"][0]
        print(" ".join(toks[sa["start_token"] : sa["end_token"]])[:500])
    else:
        print("No short answer.")



## === cell 10
if "annotations" in train.columns:
    train["D"] = [t[0]["long_answer"]["start_token"] for t in train.annotations]
    print(train["D"].head())



## === cell 11
if len(train) > 5:
    i = 5
    ann = train.annotations.iloc[i][0]
    toks = train.document_text.iloc[i].split()

    print(train.annotations.iloc[i])
    print(train.question_text.iloc[i])
    print("Long Answer:")
    la = ann["long_answer"]
    if la["start_token"] >= 0 and la["end_token"] > la["start_token"]:
        print(" ".join(toks[la["start_token"] : la["end_token"]])[:500])
    else:
        print("No long answer.")
    print("Short Answer:")
    if len(ann.get("short_answers", [])) > 0:
        sa = ann["short_answers"][0]
        print(" ".join(toks[sa["start_token"] : sa["end_token"]])[:500])
    else:
        print("No short answer.")



## === cell 12
if "D" in train.columns:
    train = train[train["D"] > -1].reset_index(drop=True)
    print("Filtered train shape:", train.shape)



## === cell 13
sub = pd.read_csv(os.path.join(p, "sample_submission.csv"))
print("train/sub:", train.shape, sub.shape)
print(sub.head())



## === cell 14
if len(train) > 99:
    i = 99
    print("URL:", train.document_url.iloc[i])
    print(train.question_text.iloc[i])
    print(train.long_answer_candidates.iloc[i][0])
    ann = train.annotations.iloc[i][0]
    toks = train.document_text.iloc[i].split()

    print("Long Answer")
    la = ann["long_answer"]
    if la["start_token"] >= 0 and la["end_token"] > la["start_token"]:
        print(" ".join(toks[la["start_token"] : la["end_token"]])[:500])
    if len(ann.get("short_answers", [])) > 0:
        print("Short Answer")
        sa = ann["short_answers"][0]
        print(" ".join(toks[sa["start_token"] : sa["end_token"]])[:200])



## === cell 15
if "annotations" in train.columns and len(train) > 0:
    la = [
        t[0]["long_answer"]["end_token"] - t[0]["long_answer"]["start_token"]
        for t in train.annotations
    ]
    sa = [
        t[0]["short_answers"][0]["end_token"] - t[0]["short_answers"][0]["start_token"]
        for t in train.annotations
        if len(t[0].get("short_answers", [])) > 0
    ]
    print(
        "Median long/short lengths:",
        np.median(la),
        (np.median(sa) if len(sa) else None),
    )



## === cell 16
import nltk
from nltk.corpus import stopwords
from html.parser import HTMLParser

try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

STOPWORDS_EN = set(stopwords.words("english"))




## === cell 17
class _PExtractor(HTMLParser):
    __slots__ = ("_in_p", "_buf", "paras")

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._in_p = False
        self._buf = []
        self.paras = []

    def handle_starttag(self, tag, attrs):
        if tag == "p":
            self._in_p = True
            self._buf.clear()

    def handle_endtag(self, tag):
        if tag == "p" and self._in_p:
            txt = "".join(self._buf).strip()
            if txt:
                self.paras.append(txt)
            self._in_p = False
            self._buf.clear()

    def handle_data(self, data):
        if self._in_p and data:
            self._buf.append(data)


def extract_paras_fast(doc: str):
    parser = _PExtractor()
    parser.feed(doc)
    return [t for t in parser.paras if len(t) > 50]


def qa_word_match(q, a_list):
    q_words = q.lower().split()
    q_words = [w for w in q_words if w not in STOPWORDS_EN]

    best = a_list[0] if len(a_list) else ""
    tm = -1
    q_set = set(q_words)
    for a in a_list:
        m = sum(1 for w in a.lower().split() if w in q_set)
        if m > tm:
            tm = m
            best = str(a)
    return best


def qa_best_overlap(q, a_list):
    q_words = q.lower().split()
    q_words = [w for w in q_words if w not in STOPWORDS_EN]
    if not q_words or not a_list:
        return -1
    q_set = set(q_words)
    best_m = -1
    for a in a_list:
        m = sum(1 for w in a.lower().split() if w in q_set)
        if m > best_m:
            best_m = m
    return best_m


def micro_f1_from_counts(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else (2 * tp) / denom


def is_non_null_span(span_str: str):
    return isinstance(span_str, str) and (":" in span_str) and len(span_str) >= 3


def _any_gold_long_answer_spans(annotations):
    spans = []
    try:
        for ann in (annotations or [])[:5]:
            la = ann.get("long_answer", {})
            s = int(la.get("start_token", -1))
            e = int(la.get("end_token", -1))
            if s >= 0 and e > s:
                spans.append((s, e))
    except Exception:
        return []
    return spans


_STRIP_CHARS = ".,;:!?\"'()[]{}<>|`~@#$%^&*-_=+\\/"


def _norm_tok(w: str) -> str:
    return w.strip(_STRIP_CHARS).lower()


def best_candidate_by_question_overlap(q: str, doc_tokens, candidates):
    q_words = [_norm_tok(w) for w in q.lower().split()]
    q_words = [w for w in q_words if w and (w not in STOPWORDS_EN)]
    if not q_words:
        return None
    q_set = set(q_words)
    best = None
    best_m = -1

    for c in candidates or []:
        if not c.get("top_level", True):
            continue
        cs = int(c.get("start_token", -1))
        ce = int(c.get("end_token", -1))
        if cs < 0 or ce <= cs or ce > len(doc_tokens):
            continue
        cand_words = doc_tokens[cs:ce]
        m = 0
        for w in cand_words:
            nw = _norm_tok(w)
            if nw and (nw in q_set):
                m += 1
        if m > best_m:
            best_m = m
            best = (cs, ce, best_m)

    return best  # (start, end, match_count) or None


def best_candidate_by_paragraph_overlap(chosen_para: str, doc_tokens, candidates):
    if not chosen_para:
        return None
    p_words = [_norm_tok(w) for w in chosen_para.lower().split()]
    p_words = [w for w in p_words if w and (w not in STOPWORDS_EN)]
    if not p_words:
        return None
    p_set = set(p_words)

    best = None
    best_m = -1
    for c in candidates or []:
        if not c.get("top_level", True):
            continue
        cs = int(c.get("start_token", -1))
        ce = int(c.get("end_token", -1))
        if cs < 0 or ce <= cs or ce > len(doc_tokens):
            continue
        cand_words = doc_tokens[cs:ce]
        m = 0
        for w in cand_words:
            nw = _norm_tok(w)
            if nw and (nw in p_set):
                m += 1
        if m > best_m:
            best_m = m
            best = (cs, ce, best_m)
    return best


def train_tune_overlap_threshold_exact(train_df, thr_candidates=(1, 2, 3, 4, 5, 6, 7)):
    if train_df is None or len(train_df) == 0 or "annotations" not in train_df.columns:
        return 2

    best_overlaps = []
    exact_match_if_predict = []

    for i in range(len(train_df)):
        try:
            row = train_df.iloc[i]
            doc = row.document_text
            toks = doc.split()
            n_toks = len(toks)

            paras = extract_paras_fast(doc)
            chosen = qa_word_match(row.question_text, paras) if paras else ""
            best_ol = qa_best_overlap(row.question_text, paras) if paras else -1

            if chosen:
                char_pos = doc.find(chosen)
            else:
                char_pos = -1

            if char_pos < 0:
                start_tok_guess = 0
            else:
                start_tok_guess = max(0, len(doc[:char_pos].split()) - 1)

            start_tok = start_tok_guess
            end_tok = min(n_toks, start_tok_guess + 114)

            cands = row.get("long_answer_candidates", [])
            best_cand = best_candidate_by_paragraph_overlap(chosen, toks, cands)
            if best_cand is None:
                best_cand = best_candidate_by_question_overlap(
                    row.question_text, toks, cands
                )

            if best_cand is not None:
                cs, ce, _m = best_cand
                start_tok, end_tok = cs, ce
            else:
                snapped = False
                for c in cands:
                    if not c.get("top_level", True):
                        continue
                    cs = int(c.get("start_token", -1))
                    ce = int(c.get("end_token", -1))
                    if cs >= 0 and ce > cs:
                        if cs <= start_tok_guess < ce:
                            start_tok, end_tok = cs, ce
                            snapped = True
                            break
                if not snapped and chosen:
                    span = len(chosen.split())
                    if span <= 0:
                        span = 114
                    end_tok = min(n_toks, start_tok + span + 2)

            gold_spans = _any_gold_long_answer_spans(row.annotations)
            is_exact = False
            if gold_spans:
                is_exact = any(
                    (start_tok == gs and end_tok == ge) for gs, ge in gold_spans
                )

            best_overlaps.append(int(best_ol))
            exact_match_if_predict.append(bool(is_exact))
        except Exception:
            best_overlaps.append(-1)
            exact_match_if_predict.append(False)

    best_thr = 2
    best_f1 = -1.0
    for thr in thr_candidates:
        tp = fp = fn = 0
        for s, is_exact in zip(best_overlaps, exact_match_if_predict):
            pred = s >= thr
            if pred and is_exact:
                tp += 1
            elif pred and (not is_exact):
                fp += 1
            elif (not pred) and is_exact:
                fn += 1
        f1 = micro_f1_from_counts(tp, fp, fn)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    return int(best_thr)


OVERLAP_THR = train_tune_overlap_threshold_exact(
    train, thr_candidates=(1, 2, 3, 4, 5, 6, 7)
)
print(
    "Tuned overlap threshold (exact-span proxy, any gold long) for predicting an answer:",
    OVERLAP_THR,
)



## === cell 18
yesno_triggers = {
    "am",
    "are",
    "can",
    "could",
    "did",
    "do",
    "does",
    "has",
    "have",
    "is",
    "may",
    "should",
    "was",
    "were",
    "will",
}

pred_long = {}
pred_short = {}

test_path = os.path.join(p, "simplified-nq-test.jsonl")
with open(test_path, "r") as f:
    for line in f:
        row = json.loads(line)
        ex_id = str(row["example_id"])
        doc = row["document_text"]
        q = row["question_text"]

        toks = doc.split()
        n_toks = len(toks)

        paras = extract_paras_fast(doc)

        best_overlap = qa_best_overlap(q, paras)
        predict_answer = best_overlap >= OVERLAP_THR

        if not predict_answer:
            pred_long[ex_id] = ""
            pred_short[ex_id] = ""
            continue

        chosen = ""
        if len(paras) > 0:
            chosen = qa_word_match(q, paras)

        if chosen:
            char_pos = doc.find(chosen)
        else:
            char_pos = -1

        if char_pos < 0:
            start_tok_guess = 0
        else:
            start_tok_guess = max(0, len(doc[:char_pos].split()) - 1)

        start_tok = start_tok_guess
        end_tok = min(n_toks, start_tok_guess + 114)

        try:
            cands = row.get("long_answer_candidates", [])

            best_cand = best_candidate_by_paragraph_overlap(chosen, toks, cands)
            if best_cand is None:
                best_cand = best_candidate_by_question_overlap(q, toks, cands)

            if best_cand is not None:
                cs, ce, _m = best_cand
                start_tok, end_tok = cs, ce
            else:
                snapped = False
                for c in cands:
                    if not c.get("top_level", True):
                        continue
                    cs = int(c.get("start_token", -1))
                    ce = int(c.get("end_token", -1))
                    if cs >= 0 and ce > cs:
                        if cs <= start_tok_guess < ce:
                            start_tok, end_tok = cs, ce
                            snapped = True
                            break
                if not snapped and chosen:
                    span = len(chosen.split())
                    if span <= 0:
                        span = 114
                    end_tok = min(n_toks, start_tok + span + 2)
        except Exception:
            if len(paras) > 0:
                if char_pos < 0:
                    start_tok = 0
                else:
                    start_tok = max(0, len(doc[:char_pos].split()) - 1)
                span = len(chosen.split()) if chosen else 0
                if span <= 0:
                    span = 114
                end_tok = min(n_toks, start_tok + span + 2)
            else:
                if n_toks <= 120:
                    start_tok = 0
                else:
                    start_tok = random.randrange(0, max(1, n_toks - 115))
                end_tok = min(n_toks, start_tok + 114)

        pred_long[ex_id] = f"{start_tok}:{end_tok}"

        q_words = set(q.lower().split())
        if len(yesno_triggers.intersection(q_words)) > 0:
            pred_short[ex_id] = random.choice(["YES", "NO"])
        else:
            if end_tok - start_tok <= 3:
                s0 = start_tok
            else:
                s0 = random.randrange(start_tok, max(start_tok + 1, end_tok - 2))
            s1 = min(n_toks, s0 + 2)
            pred_short[ex_id] = f"{s0}:{s1}"

out = sub.copy()


def base_id_and_type(example_id_with_suffix: str):
    if example_id_with_suffix.endswith("_long"):
        return example_id_with_suffix[:-5], "long"
    if example_id_with_suffix.endswith("_short"):
        return example_id_with_suffix[:-6], "short"
    return example_id_with_suffix, None


eids = out["example_id"].astype(str).to_numpy()
preds = []
append = preds.append
for eid in eids:
    base, typ = base_id_and_type(eid)
    if typ == "long":
        append(pred_long.get(base, ""))
    elif typ == "short":
        append(pred_short.get(base, ""))
    else:
        append("")

out["PredictionString"] = (
    pd.Series(preds, index=out.index).astype(str).replace({"nan": ""})
)
out.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", out.shape)
print(out.head())
