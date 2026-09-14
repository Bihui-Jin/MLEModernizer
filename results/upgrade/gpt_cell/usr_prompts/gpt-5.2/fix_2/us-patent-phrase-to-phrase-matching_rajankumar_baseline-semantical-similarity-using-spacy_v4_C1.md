# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.10

# 2. Installed packages

No external packages required in the script and installed.

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        input/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        working/
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
```

-> data/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> data/us-patent-phrase-to-phrase-matching/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/us-patent-phrase-to-phrase-matching/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/us-patent-phrase-to-phrase-matching/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> input/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
import os
import sys
import pandas as pd
import spacy
import time


## === cell 1
TRAIN_FILE_PATH = '../input/us-patent-phrase-to-phrase-matching/train.csv'
TEST_FILE_PATH = '../input/us-patent-phrase-to-phrase-matching/test.csv'
SAMPLE_SUBMISSION_PATH = '../input/us-patent-phrase-to-phrase-matching/sample_submission.csv'


## === cell 2
class config:
    PRINT_EVERY_N_WORD = 100
    BAR_LEN = 50


## === cell 3
train_df = pd.read_csv(TRAIN_FILE_PATH)
test_df = pd.read_csv(TEST_FILE_PATH)
submission_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)

print('train_df shape:', train_df.shape)
print('test_df shape:', test_df.shape)
print('submission_df shape:', submission_df.shape)


## === cell 4
similarity_score = []
n_words = train_df.shape[0]
start = time.time()

try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    nlp = spacy.blank("en")

    def _fallback_similarity(doc1, doc2):
        s1 = {t.text.lower() for t in doc1 if not t.is_space}
        s2 = {t.text.lower() for t in doc2 if not t.is_space}
        if not s1 and not s2:
            return 0.0
        return len(s1 & s2) / len(s1 | s2)

    nlp.vocab.vectors_length = 0  # ensure we don't rely on missing vectors
    from spacy.tokens import Doc

    Doc.similarity = lambda self, other: float(_fallback_similarity(self, other))

for i, row in train_df.iterrows():
    token1 = nlp(row.anchor)
    token2 = nlp(row.target)
    similarity_score.append(token1.similarity(token2))

    if ((i + 1) % config.PRINT_EVERY_N_WORD == 0) | (i + 1 == n_words):
        end = time.time()
        time_elapsed = end - start
        if i + 1 == n_words:
            bar = (
                "["
                + "=" * int((i + 1) * config.BAR_LEN / n_words)
                + "." * (config.BAR_LEN - int((i + 1) * config.BAR_LEN / n_words) - 1)
                + "]"
            )
        else:
            bar = (
                "["
                + "=" * int((i + 1) * config.BAR_LEN / n_words)
                + ">"
                + "." * (config.BAR_LEN - int((i) * config.BAR_LEN / n_words) - 1)
                + "]"
            )
        perc = (i + 1) * 100 / n_words
        sys.stdout.write("\r")
        sys.stdout.write(
            "%i/%i words completed %s %d%% %.1fs %.1fms/word"
            % (i + 1, n_words, bar, perc, time_elapsed, time_elapsed * 1000 / (i + 1))
        )
        sys.stdout.flush()

train_df["similarity_score"] = similarity_score


## --- ERROR in cell 4, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mOSError[0m                                   Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2349841380.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      8[0m [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 9[0;31m     [0mnlp[0m [0;34m=[0m [0mspacy[0m[0;34m.[0m[0mload[0m[0;34m([0m[0;34m"en_core_web_lg"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     10[0m [0;32mexcept[0m [0mOSError[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/spacy/__init__.py[0m in [0;36mload[0;34m(name, vocab, disable, enable, exclude, config)[0m
[1;32m     51[0m     """
[0;32m---> 52[0;31m     return util.load_model(
[0m[1;32m     53[0m         [0mname[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/spacy/util.py[0m in [0;36mload_model[0;34m(name, vocab, disable, enable, exclude, config)[0m
[1;32m    483[0m         [0;32mraise[0m [0mIOError[0m[0;34m([0m[0mErrors[0m[0;34m.[0m[0mE941[0m[0;34m.[0m[0mformat[0m[0;34m([0m[0mname[0m[0;34m=[0m[0mname[0m[0;34m,[0m [0mfull[0m[0;34m=[0m[0mOLD_MODEL_SHORTCUTS[0m[0;34m[[0m[0mname[0m[0;34m][0m[0;34m)[0m[0;34m)[0m  [0;31m# type: ignore[index][0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 484[0;31m     [0;32mraise[0m [0mIOError[0m[0;34m([0m[0mErrors[0m[0;34m.[0m[0mE050[0m[0;34m.[0m[0mformat[0m[0;34m([0m[0mname[0m[0;34m=[0m[0mname[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    485[0m [0;34m[0m[0m

[0;31mOSError[0m: [E050] Can't find model 'en_core_web_lg'. It doesn't seem to be a Python package or a valid path to a data directory.

During handling of the above exception, another exception occurred:

[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2349841380.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     20[0m [0;34m[0m[0m
[1;32m     21[0m     [0;31m# Monkey-patch Doc.similarity for this pipeline only (keeps calling code unchanged)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 22[0;31m     [0mnlp[0m[0;34m.[0m[0mvocab[0m[0;34m.[0m[0mvectors_length[0m [0;34m=[0m [0;36m0[0m  [0;31m# ensure we don't rely on missing vectors[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     23[0m     [0;32mfrom[0m [0mspacy[0m[0;34m.[0m[0mtokens[0m [0;32mimport[0m [0mDoc[0m[0;34m[0m[0;34m[0m[0m
[1;32m     24[0m [0;34m[0m[0m

[0;31mAttributeError[0m: attribute 'vectors_length' of 'spacy.vocab.Vocab' objects is not writable

## === cell 5
'''
0.000 - 0.125 -> 0.00
0.125 - 0.375 -> 0.25
0.375 - 0.625 -> 0.50
0.625 - 0.875 -> 0.75
0.875 - 1.000 -> 1.00
'''

mapping = {0.00: [0.000, 0.125],
           0.25: [0.125, 0.375],
           0.50: [0.375, 0.625],
           0.75: [0.625, 0.875],
           1.00: [0.875, 1.000]}

for key in mapping.keys():
    train_df['similarity_score'] = train_df['similarity_score'].mask((train_df['similarity_score'] >= mapping[key][0]) & (train_df['similarity_score'] < mapping[key][1]), key)
